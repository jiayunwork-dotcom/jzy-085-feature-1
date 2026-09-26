"""弹性稳定（分岔屈曲）分析：与静力核算并列的第二种分析类型。

分两段走，两段之间的数据严格同源：

第一段（一阶线性分析）：直接复用 analysis.run_linear_analysis ——
    同一套校验、同一套组装、同一种划行划列约束处理、同一个求解器，
    得到当前荷载下每根杆的真实轴力（拉力为正）。
    几何 / 荷载非法、结构是机构等情形在这一段按原有错误代码抛出。

第二段（广义特征值）：按各杆轴力逐杆形成一致几何刚度
    （element.local_geometric_stiffness），用与弹性刚度完全相同的
    坐标变换（geometry.stiffness_to_global）与 scatter-add 组装
    （assembly.assemble_matrix）形成整体几何刚度，并在同一组自由
    自由度上用同一个 condense_matrix 缩聚，最后求解

        K_e φ = λ (−K_g) φ

    最低的正特征值 λ_cr 即弹性临界荷载因子（当前荷载整体放大 λ_cr
    倍时结构发生分岔失稳），对应特征向量为屈曲模态。
    不存在正特征值（受拉主导 / 无轴力）时抛 NoBucklingModeError，
    明确给出「该荷载方向下不发生弹性屈曲」的结论。
"""

from __future__ import annotations

import numpy as np

from . import assembly, element, geometry
from .analysis import run_linear_analysis
from .constraints import condense_matrix
from .eigen import lowest_positive_factor
from .errors import EigenSolutionError, NoBucklingModeError


def assemble_geometric_stiffness(
    ndof: int,
    members: list,
    geometries: dict,
    node_index: dict[str, int],
    axial_forces: dict[str, float],
) -> np.ndarray:
    """按各杆真实轴力组装整体几何刚度矩阵（未缩聚）。

    每根杆的几何刚度是它自身轴力的函数：轴力大小决定量级，
    拉 / 压决定加固还是削弱抗侧刚度；坐标变换与弹性刚度共用同一套约定。
    """
    kg_global_list: list[np.ndarray] = []
    for member in members:
        geom = geometries[member.id]
        kg_local = element.local_geometric_stiffness(
            geom.length, axial_forces[member.id]
        )
        t = geometry.transformation_matrix(geom)
        kg_global_list.append(geometry.stiffness_to_global(kg_local, t))
    return assembly.assemble_matrix(ndof, members, node_index, kg_global_list)


def analyze_stability(frame) -> dict:
    """对一副刚架执行弹性稳定分析，返回可序列化为 StabilityResponse 的字典。"""
    # ---- 第一段：一阶线性分析（与静力核算完全同源） ----------------------
    ctx = run_linear_analysis(frame)
    ndof = ctx["ndof"]
    free_dofs = ctx["free_dofs"]

    # ---- 第二段：几何刚度 + 广义特征值 ----------------------------------
    k_g = assemble_geometric_stiffness(
        ndof, frame.members, ctx["geometries"], ctx["node_index"], ctx["axial_forces"]
    )
    # 弹性刚度与几何刚度在同一组自由度、同一种划行划列方式下缩聚
    k_e_ff = condense_matrix(ctx["stiffness"], free_dofs)
    k_g_ff = condense_matrix(k_g, free_dofs)

    factor, mode_free = lowest_positive_factor(k_e_ff, k_g_ff)
    if factor is None:
        raise NoBucklingModeError(
            "该荷载方向下不存在正的弹性临界荷载因子："
            "结构以受拉为主或没有轴向力，不发生弹性屈曲"
        )

    # 屈曲模态：补零还原成全自由度，归一化（绝对值最大的分量定为 +1，
    # 位移与转角量纲不同，归一化仅为确定形状，幅值本身无物理意义）
    mode = np.zeros(ndof, dtype=float)
    mode[free_dofs] = mode_free
    scale = float(np.max(np.abs(mode)))
    if not np.isfinite(scale) or scale == 0.0:
        raise EigenSolutionError("屈曲模态归一化失败（特征向量为零或非有限）")
    mode /= scale
    if mode[int(np.argmax(np.abs(mode)))] < 0.0:
        mode = -mode
    mode = mode + 0.0  # 符号翻转会把零分量变成 -0.0，归一为 +0.0 再输出

    node_index = ctx["node_index"]
    mode_shape = []
    for node in frame.nodes:
        idx = node_index[node.id]
        mode_shape.append(
            {
                "node_id": node.id,
                "ux": float(mode[3 * idx]),
                "uy": float(mode[3 * idx + 1]),
                "theta": float(mode[3 * idx + 2]),
            }
        )

    return {
        "success": True,
        "has_buckling_mode": True,
        "critical_load_factor": float(factor),
        "mode_shape": mode_shape,
    }

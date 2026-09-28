"""把各功能段串成完整核算：线性静力（/solve）与弹性屈曲（/buckling）。

两个分析类型并列对外，共用同一套输入模型与校验：

- :func:`analyze_frame`：一阶线性静力分析，输出位移 / 杆端内力 / 反力
  （行为与响应结构保持不变）；
- :func:`analyze_buckling`：先完整跑一遍一阶线性分析取得每根杆的真实轴力，
  再据此逐杆形成几何刚度、按同一套划行划列约束缩聚后求解广义特征值问题，
  输出最低正临界荷载因子与屈曲模态。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import assembly, element, forces as forces_mod
from .buckling import local_geometric_stiffness, solve_buckling_eigenproblem
from .constraints import classify_dofs, condense
from .element import local_stiffness, uniform_transverse_equivalent_loads
from .geometry import force_to_global, stiffness_to_global, transformation_matrix
from .errors import SingularMatrixError
from .solver import solve_system
from .validation import validate


@dataclass
class LinearSolution:
    """第一段一阶线性分析的全部中间量，供静力输出与稳定分析复用。"""

    displacement: np.ndarray
    load: np.ndarray
    stiffness: np.ndarray
    free_dofs: np.ndarray
    restrained_dofs: np.ndarray
    node_index: dict[str, int]
    # 每根杆（按输入顺序）：(杆件模型, 几何, 局部刚度, 变换阵, 等效荷载, 整体刚度, 轴力)
    member_data: list


def run_linear_analysis(frame) -> LinearSolution:
    """一阶线性静力分析的完整第一段（校验 → 组装 → 约束 → 求解 → 轴力）。"""
    # 1) 校验（含连通性与整体刚体约束检查），顺便复用几何信息
    validation_info = validate(frame)
    geometries = validation_info["geometries"]
    node_index = validation_info["node_index"]
    n_nodes = len(frame.nodes)
    ndof = 3 * n_nodes

    # 分布荷载按杆件编号索引（当前每根杆至多一条，多条时叠加）
    udl_by_member: dict[str, float] = {}
    for dist in frame.distributed_loads:
        udl_by_member[dist.member_id] = udl_by_member.get(dist.member_id, 0.0) + dist.qy

    # 2) 逐杆：局部刚度 -> 坐标变换 -> 等效节点力
    k_global_list: list[np.ndarray] = []
    eq_global_list: list[np.ndarray] = []
    member_local: list[tuple] = []
    for member in frame.members:
        geom = geometries[member.id]
        k_local = local_stiffness(
            geom.length, member.elastic_modulus, member.area, member.inertia
        )
        t = transformation_matrix(geom)
        k_global = stiffness_to_global(k_local, t)
        k_global_list.append(k_global)

        qy = udl_by_member.get(member.id, 0.0)
        eq_local = uniform_transverse_equivalent_loads(geom.length, qy)
        eq_global_list.append(force_to_global(eq_local, t))
        member_local.append((member, geom, k_local, t, eq_local, k_global))

    # 3) 总刚与总荷载向量组装
    stiffness, load = assembly.assemble(
        ndof,
        frame.members,
        node_index,
        k_global_list,
        eq_global_list,
        frame.nodal_loads,
    )

    # 4) 划行划列施加约束，求解自由自由度
    restraints = [node.restraints for node in frame.nodes]
    free_dofs, restrained_dofs = classify_dofs(restraints)
    k_ff, p_f = condense(stiffness, load, free_dofs)
    d_free = solve_system(k_ff, p_f)  # 奇异矩阵在此抛 SingularMatrixError

    # 5) 还原完整位移向量（被约束自由度位移为零）
    displacement = np.zeros(ndof, dtype=float)
    displacement[free_dofs] = d_free
    _guard_finite(displacement, "节点位移")

    # 6) 逐杆取真实轴力（拉正压负），与静力输出同源
    member_data: list = []
    for member, geom, k_local, t, eq_local, k_global in member_local:
        ii = node_index[member.node_i]
        jj = node_index[member.node_j]
        member_dofs = np.array([3 * ii, 3 * ii + 1, 3 * ii + 2,
                                3 * jj, 3 * jj + 1, 3 * jj + 2])
        f_local = forces_mod.member_end_forces_local(
            k_local, t, displacement[member_dofs], eq_local
        )
        _guard_finite(f_local, f"杆件 {member.id} 的杆端力")
        axial = float(f_local[0])
        member_data.append((member, geom, k_local, t, eq_local, k_global, axial))

    return LinearSolution(
        displacement=displacement,
        load=load,
        stiffness=stiffness,
        free_dofs=free_dofs,
        restrained_dofs=restrained_dofs,
        node_index=node_index,
        member_data=member_data,
    )


def analyze_frame(frame) -> dict:
    """对一副刚架执行线性静力核算，返回可直接序列化为 SolveResponse 的字典。"""
    sol = run_linear_analysis(frame)

    # 6) 各杆杆端内力（局部刚度 × 局部位移）
    member_forces_out: list[dict] = []
    for member, geom, k_local, t, eq_local, _k_global, axial in sol.member_data:
        ii = sol.node_index[member.node_i]
        jj = sol.node_index[member.node_j]
        member_dofs = np.array([3 * ii, 3 * ii + 1, 3 * ii + 2,
                                3 * jj, 3 * jj + 1, 3 * jj + 2])
        f_local = forces_mod.member_end_forces_local(
            k_local, t, sol.displacement[member_dofs], eq_local
        )

        member_forces_out.append(
            {
                "member_id": member.id,
                "node_i": member.node_i,
                "node_j": member.node_j,
                "end_i": {
                    "axial": float(f_local[0]),
                    "shear": float(f_local[1]),
                    "moment": float(f_local[2]),
                },
                "end_j": {
                    "axial": float(f_local[3]),
                    "shear": float(f_local[4]),
                    "moment": float(f_local[5]),
                },
                "axial_force": axial,
            }
        )

    # 7) 支座反力回代
    reaction_vector = forces_mod.reactions(
        sol.stiffness, sol.displacement, sol.load, sol.restrained_dofs
    )
    _guard_finite(reaction_vector, "支座反力")

    reactions_out: list[dict] = []
    for node in frame.nodes:
        idx = sol.node_index[node.id]
        reactions_out.append(
            {
                "node_id": node.id,
                "fx": float(reaction_vector[3 * idx]),
                "fy": float(reaction_vector[3 * idx + 1]),
                "moment": float(reaction_vector[3 * idx + 2]),
                "restrained": [bool(r) for r in node.restraints],
            }
        )

    displacements_out = []
    for node in frame.nodes:
        idx = sol.node_index[node.id]
        displacements_out.append(
            {
                "node_id": node.id,
                "ux": float(sol.displacement[3 * idx]),
                "uy": float(sol.displacement[3 * idx + 1]),
                "theta": float(sol.displacement[3 * idx + 2]),
            }
        )

    return {
        "success": True,
        "displacements": displacements_out,
        "member_forces": member_forces_out,
        "reactions": reactions_out,
    }


def analyze_buckling(frame) -> dict:
    """弹性临界荷载因子分析，返回可序列化为 BucklingResponse 的字典。

    第一段：与静力分析完全相同的一阶线性分析（含分布荷载等效节点力），
    取得每根杆的轴力；第二段：逐杆用该轴力形成几何刚度，整体组装后在与
    弹性刚度**同一组自由自由度、同一种划行划列约束**下缩聚，求解
    ``(K_e,f + λ K_g,f) φ = 0``，报告最低正根。
    """
    sol = run_linear_analysis(frame)
    ndof = sol.stiffness.shape[0]
    geometric = np.zeros((ndof, ndof), dtype=float)

    # 第二段：逐杆几何刚度——轴力必须取自第一段真实解（含正负号）
    for member, geom, _k_local, t, _eq_local, _k_global, axial in sol.member_data:
        kg_local = local_geometric_stiffness(geom.length, axial)
        kg_global = t.T @ kg_local @ t  # 与弹性刚度共用同一套坐标变换约定
        ii = sol.node_index[member.node_i]
        jj = sol.node_index[member.node_j]
        member_dofs = np.array([3 * ii, 3 * ii + 1, 3 * ii + 2,
                                3 * jj, 3 * jj + 1, 3 * jj + 2])
        geometric[np.ix_(member_dofs, member_dofs)] += kg_global

    _guard_finite(geometric, "整体几何刚度")

    # 同组自由度、同种约束消除方式缩聚到一起（绝不与罚函数混用）
    k_ff = sol.stiffness[np.ix_(sol.free_dofs, sol.free_dofs)]
    kg_ff = geometric[np.ix_(sol.free_dofs, sol.free_dofs)]
    lambda_cr, phi_free = solve_buckling_eigenproblem(k_ff, kg_ff)

    # 还原全自由度模态（被约束自由度置零），再做无量纲归一化：
    # 最大绝对分量取 1，且最大分量为正，消除特征向量符号歧义
    mode_full = np.zeros(ndof, dtype=float)
    mode_full[sol.free_dofs] = phi_free
    _guard_finite(mode_full, "屈曲模态")
    max_abs = float(np.max(np.abs(mode_full)))
    if max_abs <= 0.0:
        raise SingularMatrixError("屈曲模态为零向量")
    peak = int(np.argmax(np.abs(mode_full)))
    if mode_full[peak] < 0.0:
        mode_full = -mode_full
    mode_full = mode_full / max_abs

    mode_out = []
    for node in frame.nodes:
        idx = sol.node_index[node.id]
        mode_out.append(
            {
                "node_id": node.id,
                "ux": float(mode_full[3 * idx]),
                "uy": float(mode_full[3 * idx + 1]),
                "theta": float(mode_full[3 * idx + 2]),
            }
        )

    return {
        "success": True,
        "has_valid_buckling_mode": True,
        "critical_load_factor": float(lambda_cr),
        "buckling_mode": mode_out,
    }


def _guard_finite(values: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(values)):
        # 正常路径下 solver 已拦截；这里保证任何阶段都不会吐出 NaN
        raise RuntimeError(f"{label}中出现非有限值（NaN 或无穷）")

"""把各功能段串成完整的核算：一阶线性静力分析 + 弹性稳定（屈曲）分析。

两个分析类型共用同一条一阶管线（校验 -> 组装 -> 划行划列 -> 求解），
保证稳定分析的几何刚度严格来自第一段真实解出的轴力，且弹性刚度与
几何刚度在同一组自由度、同一种约束消除方式下缩聚：

- :func:`analyze_frame`    —— 线性静力核算（位移 / 内力 / 反力）；
- :func:`analyze_buckling` —— 弹性稳定分析（临界荷载因子 + 屈曲模态）。
"""

from __future__ import annotations

import numpy as np

from . import assembly, forces as forces_mod
from .constraints import classify_dofs, condense
from .element import local_stiffness, uniform_transverse_equivalent_loads
from .geometry import force_to_global, stiffness_to_global, transformation_matrix
from .solver import solve_system
from .stability import (
    local_geometric_stiffness,
    normalize_mode_shape,
    solve_critical_load,
)
from .validation import validate


def _assemble_validated_system(frame) -> dict:
    """一阶管线前半：校验模型，组装总刚与总荷载向量，分类自由度。"""
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
    local_data: list[tuple] = []
    for member in frame.members:
        geom = geometries[member.id]
        k_local = local_stiffness(
            geom.length, member.elastic_modulus, member.area, member.inertia
        )
        t = transformation_matrix(geom)
        k_global_list.append(stiffness_to_global(k_local, t))

        qy = udl_by_member.get(member.id, 0.0)
        eq_local = uniform_transverse_equivalent_loads(geom.length, qy)
        eq_global_list.append(force_to_global(eq_local, t))
        local_data.append((member, geom, k_local, t, eq_local))

    # 3) 总刚与总荷载向量组装
    stiffness, load = assembly.assemble(
        ndof,
        frame.members,
        node_index,
        k_global_list,
        eq_global_list,
        frame.nodal_loads,
    )

    # 4) 自由度分类（划行划列法，静力与稳定共用同一组自由 / 约束自由度）
    restraints = [node.restraints for node in frame.nodes]
    free_dofs, restrained_dofs = classify_dofs(restraints)

    return {
        "node_index": node_index,
        "ndof": ndof,
        "local_data": local_data,
        "stiffness": stiffness,
        "load": load,
        "free_dofs": free_dofs,
        "restrained_dofs": restrained_dofs,
    }


def _solve_first_order(system: dict) -> np.ndarray:
    """一阶管线后半：缩聚求解自由自由度，还原完整位移向量。"""
    k_ff, p_f = condense(system["stiffness"], system["load"], system["free_dofs"])
    d_free = solve_system(k_ff, p_f)  # 奇异矩阵在此抛 SingularMatrixError

    displacement = np.zeros(system["ndof"], dtype=float)
    displacement[system["free_dofs"]] = d_free
    _guard_finite(displacement, "节点位移")
    return displacement


def _member_axial_forces(system: dict, displacement: np.ndarray) -> dict[str, float]:
    """由一阶位移回代每根杆的轴力（拉力为正），与静力入口完全同一算法。"""
    node_index = system["node_index"]
    axial_forces: dict[str, float] = {}
    for member, geom, k_local, t, eq_local in system["local_data"]:
        ii = node_index[member.node_i]
        jj = node_index[member.node_j]
        member_dofs = np.array([3 * ii, 3 * ii + 1, 3 * ii + 2,
                                3 * jj, 3 * jj + 1, 3 * jj + 2])
        f_local = forces_mod.member_end_forces_local(
            k_local, t, displacement[member_dofs], eq_local
        )
        _guard_finite(f_local, f"杆件 {member.id} 的杆端力")
        # f_local 为“杆作用于节点”：i 端轴向分量即轴力（拉正）
        axial_forces[member.id] = float(f_local[0])
    return axial_forces


def analyze_frame(frame) -> dict:
    """对一副刚架执行完整核算，返回可直接序列化为 SolveResponse 的字典。"""
    system = _assemble_validated_system(frame)
    displacement = _solve_first_order(system)
    node_index = system["node_index"]

    # 各杆杆端内力（局部刚度 × 局部位移）
    axial_forces = _member_axial_forces(system, displacement)
    member_forces_out: list[dict] = []
    for member, geom, k_local, t, eq_local in system["local_data"]:
        ii = node_index[member.node_i]
        jj = node_index[member.node_j]
        member_dofs = np.array([3 * ii, 3 * ii + 1, 3 * ii + 2,
                                3 * jj, 3 * jj + 1, 3 * jj + 2])
        f_local = forces_mod.member_end_forces_local(
            k_local, t, displacement[member_dofs], eq_local
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
                "axial_force": axial_forces[member.id],
            }
        )

    # 支座反力回代
    reaction_vector = forces_mod.reactions(
        system["stiffness"], displacement, system["load"], system["restrained_dofs"]
    )
    _guard_finite(reaction_vector, "支座反力")

    reactions_out: list[dict] = []
    for node in frame.nodes:
        idx = node_index[node.id]
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
        idx = node_index[node.id]
        displacements_out.append(
            {
                "node_id": node.id,
                "ux": float(displacement[3 * idx]),
                "uy": float(displacement[3 * idx + 1]),
                "theta": float(displacement[3 * idx + 2]),
            }
        )

    return {
        "success": True,
        "displacements": displacements_out,
        "member_forces": member_forces_out,
        "reactions": reactions_out,
    }


def analyze_buckling(frame) -> dict:
    """弹性稳定分析：临界荷载因子 + 屈曲模态。

    分两段走，与静力核算共用同一条一阶管线：

    1. 一阶线性分析，解出当前荷载下每根杆的真实轴力；
    2. 逐杆按轴力形成几何刚度（与弹性刚度共用同一套坐标变换），
       组装成整体几何刚度，与弹性刚度在**同一组自由自由度**上缩聚，
       求解广义特征值问题 (K_e + λ K_g) φ = 0，取最低正特征值。

    不存在正临界因子（受拉主导 / 零荷载）时抛 NoBucklingModeError。
    """
    # 第一段：与静力入口完全相同的一阶分析（校验、组装、约束、求解）
    system = _assemble_validated_system(frame)
    displacement = _solve_first_order(system)

    # 第一段产出的真实轴力（拉力为正）
    axial_forces = _member_axial_forces(system, displacement)

    # 第二段：逐杆几何刚度 -> 同一套坐标变换 -> 同一套自由度组装
    kg_global_list: list[np.ndarray] = []
    for member, geom, k_local, t, eq_local in system["local_data"]:
        kg_local = local_geometric_stiffness(geom.length, axial_forces[member.id])
        kg_global_list.append(stiffness_to_global(kg_local, t))

    geometric = assembly.assemble_matrix(
        system["ndof"], frame.members, system["node_index"], kg_global_list
    )
    _guard_finite(geometric, "整体几何刚度矩阵")

    # 弹性刚度与几何刚度在同一组自由度、同一种划行划列方式下缩聚
    free_dofs = system["free_dofs"]
    k_ff = system["stiffness"][np.ix_(free_dofs, free_dofs)]
    kg_ff = geometric[np.ix_(free_dofs, free_dofs)]

    lambda_cr, phi_free = solve_critical_load(k_ff, kg_ff)

    # 屈曲模态还原到全部自由度（被约束自由度为零）并归一化
    mode = np.zeros(system["ndof"], dtype=float)
    mode[free_dofs] = phi_free
    mode = normalize_mode_shape(mode)
    _guard_finite(mode, "屈曲模态")

    node_index = system["node_index"]
    mode_out = []
    for node in frame.nodes:
        idx = node_index[node.id]
        mode_out.append(
            {
                "node_id": node.id,
                "ux": float(mode[3 * idx]),
                "uy": float(mode[3 * idx + 1]),
                "theta": float(mode[3 * idx + 2]),
            }
        )

    return {
        "success": True,
        "critical_load_factor": float(lambda_cr),
        "has_buckling_mode": True,
        "buckling_mode": mode_out,
    }


def _guard_finite(values: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(values)):
        # 正常路径下 solver 已拦截；这里保证任何阶段都不会吐出 NaN
        raise RuntimeError(f"{label}中出现非有限值（NaN 或无穷）")

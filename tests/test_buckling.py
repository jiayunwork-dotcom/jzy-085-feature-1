"""弹性屈曲（临界荷载因子）分析测试。

覆盖需求点：

1. 欧拉闭式解：两端铰接（计算长度 L）与一端固定一端自由（计算长度 2L）
   等截面直轴心压杆，临界因子 × 施加轴压命中 π²EI/L₀²（具体数值容差），
   且离散加密误差单调收敛；
2. 受拉主导结构 / 零荷载报 NO_POSITIVE_CRITICAL_FACTOR，不乱给数；
3. 荷载整体缩放：P 翻倍 → 临界因子减半（临界的是荷载总量）；
4. 几何刚度在纯拉 / 纯压下对抗侧刚度的加减号方向正确
   （二次型：拉正压负，对称变换后符号保持）；
5. 屈曲模态形状的物理与数值性质（边界零、归一化、对称半波）；
6. 非法输入 / 机构在 /buckling 上同样返回结构化错误、不出 NaN。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import make_axial_column, make_portal_frame
from framesolver import analyze_buckling
from framesolver.buckling import local_geometric_stiffness
from framesolver.errors import NoBucklingError
from framesolver.geometry import frame_geometry, stiffness_to_global, transformation_matrix

# 与欧拉闭式解比较的具体容差
TOL_EULER_COARSE = 1.0e-2   # 粗网格（2 段）1%
TOL_EULER_FINE = 2.0e-3     # 细网格（4 段起）0.2%
TOL_SCALING = 1.0e-9        # 荷载反比例：机器精度级
TOL_SIGN = 1.0e-12          # 几何刚度二次型符号
TOL_FINITE = 1.0e-12


# ---------------------------------------------------------------------------
# 1) 欧拉闭式解 + 收敛
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "boundary,effective_factor,n_segments,rel_tol",
    [
        ("pinned-pinned", 1.0, 2, TOL_EULER_COARSE),
        ("pinned-pinned", 1.0, 4, TOL_EULER_FINE),
        ("pinned-pinned", 1.0, 8, 5.0e-4),
        ("pinned-pinned", 1.0, 16, 1.0e-4),
        ("fixed-free", 2.0, 1, TOL_EULER_COARSE),
        ("fixed-free", 2.0, 2, TOL_EULER_FINE),
        ("fixed-free", 2.0, 4, 5.0e-4),
        ("fixed-free", 2.0, 8, 1.0e-4),
    ],
)
def test_critical_load_matches_euler_formula(boundary, effective_factor,
                                             n_segments, rel_tol):
    """λ_cr × P 与欧拉临界力 π²EI/(kL)² 的相对误差落在具体容差内。"""
    frame, p = make_axial_column(n_segments=n_segments, boundary=boundary)
    result = analyze_buckling(frame)

    assert result["success"] is True
    assert result["has_valid_buckling_mode"] is True
    lam = result["critical_load_factor"]
    assert math.isfinite(lam) and lam > 0.0

    p_critical = lam * p["p"]
    assert p_critical == pytest.approx(p["euler_load"], rel=rel_tol), (
        f"{boundary} nel={n_segments}: Pcr={p_critical}, Euler={p['euler_load']}"
    )


@pytest.mark.parametrize("boundary", ["pinned-pinned", "fixed-free"])
def test_error_converges_with_mesh_refinement(boundary):
    """分段变细，与欧拉解的相对误差应单调（或总体）下降到 1e-4 以下。"""
    errors = []
    for n_segments in (2, 4, 8, 16):
        frame, p = make_axial_column(n_segments=n_segments, boundary=boundary)
        lam = analyze_buckling(frame)["critical_load_factor"]
        rel = abs(lam * p["p"] / p["euler_load"] - 1.0)
        errors.append(rel)
    # 最粗网格误差最大、最细网格误差最小，且误差逐级下降（一致几何刚度
    # 从上侧单调收敛，这里允许末两级的数值平台）
    assert errors[-1] < errors[0]
    assert errors[0] > 5.0e-4
    assert errors[-1] < 1.0e-4
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < coarse * 0.25  # 每加密一倍误差至少缩小 4 倍


def test_euler_effective_length_doubles_for_fixed_free():
    """悬臂柱（计算长度 2L）临界力为铰接柱（L）的四分之一；
    同一施加轴压下临界因子同样为 1/4。"""
    pinned, _ = make_axial_column(n_segments=8, boundary="pinned-pinned")
    free, _ = make_axial_column(n_segments=8, boundary="fixed-free")
    lam_p = analyze_buckling(pinned)["critical_load_factor"]
    lam_f = analyze_buckling(free)["critical_load_factor"]
    assert lam_f / lam_p == pytest.approx(0.25, rel=1.0e-3)


# ---------------------------------------------------------------------------
# 2) 受拉主导 / 零荷载：无正临界因子
# ---------------------------------------------------------------------------


def test_tension_dominated_reports_no_positive_factor():
    """同一根柱反向加载（整体受拉），几何刚度处处加固，不存在正屈曲因子。"""
    frame, _ = make_axial_column(n_segments=4, boundary="pinned-pinned")
    frame.nodal_loads[0].fy = +10.0  # 反向：顶端受拉
    with pytest.raises(NoBucklingError) as exc:
        analyze_buckling(frame)
    assert exc.value.code == "NO_POSITIVE_CRITICAL_FACTOR"
    assert exc.value.http_status == 422
    assert exc.value.message


def test_zero_load_reports_no_positive_factor():
    frame, _ = make_axial_column(n_segments=4, boundary="pinned-pinned")
    frame.nodal_loads = []
    with pytest.raises(NoBucklingError) as exc:
        analyze_buckling(frame)
    assert exc.value.code == "NO_POSITIVE_CRITICAL_FACTOR"


def test_tiny_compression_load_gives_finite_large_factor():
    """极小轴压下临界因子很大但必须有限，且 λ·P 仍等于临界力总量。"""
    frame, p = make_axial_column(n_segments=2, boundary="pinned-pinned", p=1.0e-9)
    result = analyze_buckling(frame)
    lam = result["critical_load_factor"]
    assert math.isfinite(lam)
    # λ·P 仍落在该网格的临界力上（2 段解比欧拉高约 0.75%）
    assert lam * p["p"] == pytest.approx(p["euler_load"], rel=1.0e-2)
    assert lam > 1.0e12


def test_fully_restrained_structure_reports_no_factor():
    """没有任何自由自由度时不能硬凑特征值。"""
    from framesolver.models import FrameInput, Member, Node, NodalLoad
    frame = FrameInput(
        nodes=[Node(id="a", x=0.0, y=0.0, restraints=[True, True, True]),
               Node(id="b", x=0.0, y=1.0, restraints=[True, True, True])],
        members=[Member(id="m", node_i="a", node_j="b",
                        elastic_modulus=1.0, area=1.0, inertia=1.0)],
        nodal_loads=[NodalLoad(node_id="b", fx=0.0, fy=-1.0, moment=0.0)],
    )
    with pytest.raises(NoBucklingError) as exc:
        analyze_buckling(frame)
    assert exc.value.code == "NO_POSITIVE_CRITICAL_FACTOR"


def test_mixed_tension_compression_still_buckles():
    """有压杆存在的混合受力结构（门式刚架竖向荷载，柱受压、梁受拉极小），
    仍应给出有限正因子，而不是被“部分受拉”误判。"""
    frame, _ = make_portal_frame(p=10.0)
    # 把水平力换成两柱顶竖向对称加载 -> 柱轴压
    from framesolver.models import NodalLoad
    frame.nodal_loads = [
        NodalLoad(node_id="2", fx=0.0, fy=-10.0, moment=0.0),
        NodalLoad(node_id="3", fx=0.0, fy=-10.0, moment=0.0),
    ]
    result = analyze_buckling(frame)
    assert result["critical_load_factor"] > 0.0
    assert math.isfinite(result["critical_load_factor"])


# ---------------------------------------------------------------------------
# 3) 荷载反比例缩放
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [1.0, 5.0, 10.0, 25.0, 100.0])
def test_critical_factor_inversely_proportional_to_load(p):
    """荷载整体缩放 α 倍，临界因子缩小为 1/α，临界总量不变。"""
    base, _ = make_axial_column(n_segments=8, boundary="pinned-pinned", p=10.0)
    scaled, ps = make_axial_column(n_segments=8, boundary="pinned-pinned", p=p)
    lam_base = analyze_buckling(base)["critical_load_factor"]
    lam_scaled = analyze_buckling(scaled)["critical_load_factor"]
    # 反比例：λ(P)·P = const
    assert lam_scaled * p == pytest.approx(lam_base * 10.0, rel=TOL_SCALING)
    assert lam_scaled == pytest.approx(lam_base * 10.0 / p, rel=TOL_SCALING)


def test_load_doubling_halves_factor():
    """需求点名的一条：荷载翻倍，临界因子减半。"""
    f1, _ = make_axial_column(n_segments=6, boundary="fixed-free", p=10.0)
    f2, _ = make_axial_column(n_segments=6, boundary="fixed-free", p=20.0)
    lam1 = analyze_buckling(f1)["critical_load_factor"]
    lam2 = analyze_buckling(f2)["critical_load_factor"]
    assert lam2 == pytest.approx(lam1 / 2.0, rel=1.0e-10)


# ---------------------------------------------------------------------------
# 4) 几何刚度符号 / 方向
# ---------------------------------------------------------------------------


def test_geometric_stiffness_sign_tension_vs_compression():
    """任意含侧移与弯曲的变形 v：拉为正（加固），压为负（削弱）。"""
    length = 4.0
    v = np.array([0.0, 1.0, 0.03, 0.0, 0.8, -0.04])
    kg_tension = local_geometric_stiffness(length, 100.0)
    kg_compression = local_geometric_stiffness(length, -100.0)

    energy_t = float(v @ kg_tension @ v)
    energy_c = float(v @ kg_compression @ v)
    assert energy_t > TOL_SIGN
    assert energy_c < -TOL_SIGN
    assert energy_c == pytest.approx(-energy_t, abs=TOL_SIGN)  # 轴力线性


def test_geometric_stiffness_axial_block():
    """纯轴向变形上只有轴向块贡献，受拉为正、受压为负（弦效应）。"""
    length = 3.0
    n_mag = 7.0
    u = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    kg = local_geometric_stiffness(length, n_mag)
    expected = n_mag / length
    assert float(u @ kg @ u) == pytest.approx(expected, abs=TOL_SIGN)
    kg_neg = local_geometric_stiffness(length, -n_mag)
    assert float(u @ kg_neg @ u) == pytest.approx(-expected, abs=TOL_SIGN)


def test_geometric_stiffness_rigid_translations_zero_energy():
    """两个单元刚体平移模态在几何刚度下零能。

    采用的是初直梁 Total Lagrangian 一致几何刚度（Green 应变
    ε = u′ + ½(v′)² 在基态线性化）。刚体转动方向上存在初应力“自旋项”
    （v'Kg v = -N·L·α²），它在结构层与随动荷载刚度配对、且随单元加密
    （单元弦转角变小）收敛到零——这是该经典格式的已知性质，不是错误；
    故这里只钉平移零能，并把自旋项的已知取值显式钉住以防符号写反。
    """
    length = 5.0
    n_comp = -12.5
    kg = local_geometric_stiffness(length, n_comp)
    translations = [
        np.array([1.0, 0, 0, 1, 0, 0]),   # 轴向平移
        np.array([0.0, 1, 0, 0, 1, 0]),   # 横向平移
    ]
    for v in translations:
        assert abs(float(v @ kg @ v)) < TOL_SIGN

    # 线性刚体转动场（θ_i=θ_j=α，v_j=Lα）的自旋项取值 N·L·α²
    alpha = 0.01
    rotation = np.array([0.0, 0.0, alpha, 0.0, length * alpha, alpha])
    spin = float(rotation @ kg @ rotation)
    assert spin == pytest.approx(n_comp * length * alpha ** 2, rel=1.0e-9)
    # 轴力反号则自旋项反号
    kg_t = local_geometric_stiffness(length, -n_comp)
    assert float(rotation @ kg_t @ rotation) == pytest.approx(
        -n_comp * length * alpha ** 2, rel=1.0e-9
    )


def test_spin_direction_is_removed_by_supports_in_assembled_structure():
    """自旋项只沿“单元整体刚体转动”方向非零；在装好支座的结构上该方向已被
    约束消除，最低屈曲根由弯曲模态给出（见欧拉收敛测试），不会被自旋污染。
    这里直接验证：铰接柱缩聚几何刚度里，弯曲屈曲根的 |δ| 远大于
    任何残留的转动型伪根，选出的模态横向能量占绝对主导。"""
    frame, _ = make_axial_column(n_segments=4, boundary="pinned-pinned")
    result = analyze_buckling(frame)
    mode = result["buckling_mode"]
    lateral = sum(m["ux"] ** 2 for m in mode)
    axial = sum(m["uy"] ** 2 for m in mode)
    assert lateral > 1.0e6 * max(axial, 1.0e-30)


def test_geometric_stiffness_uses_same_transformation_convention():
    """坐标变换必须与弹性刚度共用同一套约定：Tᵀ Kg T 保持对称、
    特征值不变（相似变换），斜杆上轴力的抗侧贡献符号方向正确。"""
    geom = frame_geometry(0.0, 0.0, 3.0, 4.0)  # 斜杆 c=0.6,s=0.8
    t = transformation_matrix(geom)
    kg = local_geometric_stiffness(geom.length, -50.0)  # 受压
    kg_global = stiffness_to_global(kg, t)
    assert np.allclose(kg_global, kg_global.T, atol=1e-12)
    # 正交变换不改变特征值谱
    assert np.allclose(np.sort(np.linalg.eigvalsh(kg)),
                       np.sort(np.linalg.eigvalsh(kg_global)), atol=1e-10)

    # 斜压杆对两端“反向侧向错动”模态应为负（削弱）；受拉为正
    # 模态：i 端 -x 小移、j 端 +x 小移，垂直杆轴方向有相对错动
    dof_global = np.array([-0.8, 0.6, 0.0, 0.8, -0.6, 0.0])
    kg_c_global = stiffness_to_global(
        local_geometric_stiffness(geom.length, -50.0), t)
    kg_t_global = stiffness_to_global(
        local_geometric_stiffness(geom.length, 50.0), t)
    assert float(dof_global @ kg_c_global @ dof_global) < -TOL_SIGN
    assert float(dof_global @ kg_t_global @ dof_global) > TOL_SIGN


def test_global_lateral_stiffness_degraded_under_compression():
    """系统级方向校验：纯受压悬臂柱的切向刚度 K_e+K_g 最低特征值，
    随轴压增大而下降（几何刚度在削弱整体抗侧），未到临界前仍为正。"""
    from framesolver.analysis import run_linear_analysis

    def reduced_matrices(frame):
        sol = run_linear_analysis(frame)
        from framesolver.buckling import local_geometric_stiffness
        ndof = sol.stiffness.shape[0]
        geom_mats = np.zeros((ndof, ndof))
        for member, geom, _kl, tr, _eq, _kg_, axial in sol.member_data:
            kg = tr.T @ local_geometric_stiffness(geom.length, axial) @ tr
            ii = sol.node_index[member.node_i]
            jj = sol.node_index[member.node_j]
            dofs = np.array([3*ii, 3*ii+1, 3*ii+2, 3*jj, 3*jj+1, 3*jj+2])
            geom_mats[np.ix_(dofs, dofs)] += kg
        free = sol.free_dofs
        return (sol.stiffness[np.ix_(free, free)],
                geom_mats[np.ix_(free, free)])

    def min_tangent_eig(frame):
        k_e, k_g = reduced_matrices(frame)
        return float(np.min(np.linalg.eigvalsh(0.5 * ((k_e + k_g) + (k_e + k_g).T))))

    f_small, _ = make_axial_column(n_segments=4, boundary="fixed-free", p=1.0)
    f_mid, _ = make_axial_column(n_segments=4, boundary="fixed-free", p=10.0)
    f_large, _ = make_axial_column(n_segments=4, boundary="fixed-free", p=50.0)
    # 轴压越大，最低切向刚度特征值越小（被削弱得越多）
    assert min_tangent_eig(f_large) < min_tangent_eig(f_mid) < min_tangent_eig(f_small)
    # 小荷载下离临界尚远，切向刚度仍正定
    assert min_tangent_eig(f_small) > 0.0


# ---------------------------------------------------------------------------
# 5) 屈曲模态形状
# ---------------------------------------------------------------------------


def test_buckling_mode_shape_pinned_column_half_sine():
    """铰接柱第一屈曲模态近似半波正弦：边界横移为零、中点横移最大。"""
    n_segments = 16
    frame, p = make_axial_column(n_segments=n_segments, boundary="pinned-pinned")
    result = analyze_buckling(frame)
    mode = {m["node_id"]: m for m in result["buckling_mode"]}

    # 边界自由度：两端 ux=0；底端 uy/theta 约束、顶端 uy 自由（但轴向分量应≈0）
    assert mode["0"]["ux"] == pytest.approx(0.0, abs=TOL_FINITE)
    assert mode[str(n_segments)]["ux"] == pytest.approx(0.0, abs=TOL_FINITE)

    xs = np.arange(n_segments + 1) / n_segments
    ux = np.array([mode[str(k)]["ux"] for k in range(n_segments + 1)])
    # 归一化：最大绝对值恰为 1，且最大分量为正
    assert np.max(np.abs(ux)) == pytest.approx(1.0, abs=1e-12)
    assert ux[np.argmax(np.abs(ux))] > 0.0
    # 与 sin(πx/L) 形状高度一致（4 段以上误差 < 0.5%）
    assert np.allclose(ux, np.sin(np.pi * xs), atol=5.0e-3)
    # 半波：全部同号、中点附近最大
    assert np.all(ux >= -1e-12)
    assert abs(ux[n_segments // 2] - 1.0) < 1.0e-6


def test_buckling_mode_shape_fixed_free_cantilever():
    """悬臂柱模态：底端三个分量全为零，顶端自由端横移最大。"""
    n_segments = 8
    frame, _ = make_axial_column(n_segments=n_segments, boundary="fixed-free")
    result = analyze_buckling(frame)
    mode = {m["node_id"]: m for m in result["buckling_mode"]}

    for comp in ("ux", "uy", "theta"):
        assert mode["0"][comp] == pytest.approx(0.0, abs=TOL_FINITE)
    ux = np.array([mode[str(k)]["ux"] for k in range(n_segments + 1)])
    assert np.max(np.abs(ux)) == pytest.approx(1.0, abs=1e-12)
    assert abs(ux[-1]) == pytest.approx(1.0, abs=1e-12)
    # 单调向自由端增大
    assert np.all(np.diff(np.abs(ux)) >= -1e-10)


def test_buckling_mode_is_eigenvector_of_tangential_stiffness():
    """模态必须满足 (K_e + λ K_g) φ ≈ 0（直接在缩聚矩阵上复核，容差具体化）。"""
    from framesolver.analysis import run_linear_analysis
    from framesolver.buckling import local_geometric_stiffness

    frame, _ = make_axial_column(n_segments=6, boundary="pinned-pinned")
    sol = run_linear_analysis(frame)
    ndof = sol.stiffness.shape[0]
    geom_mats = np.zeros((ndof, ndof))
    for member, geom, _kl, tr, _eq, _kg_, axial in sol.member_data:
        kg = tr.T @ local_geometric_stiffness(geom.length, axial) @ tr
        ii = sol.node_index[member.node_i]
        jj = sol.node_index[member.node_j]
        dofs = np.array([3*ii, 3*ii+1, 3*ii+2, 3*jj, 3*jj+1, 3*jj+2])
        geom_mats[np.ix_(dofs, dofs)] += kg

    result = analyze_buckling(frame)
    lam = result["critical_load_factor"]
    phi = np.zeros(ndof)
    for m in result["buckling_mode"]:
        idx = sol.node_index[m["node_id"]]
        phi[3*idx] = m["ux"]
        phi[3*idx+1] = m["uy"]
        phi[3*idx+2] = m["theta"]
    residual = (sol.stiffness + lam * geom_mats) @ phi
    assert np.max(np.abs(residual)) == pytest.approx(0.0, abs=1.0e-7)


def test_response_contains_no_nan():
    frame, _ = make_axial_column(n_segments=4, boundary="pinned-pinned")
    result = analyze_buckling(frame)
    assert result["success"] is True
    assert result["has_valid_buckling_mode"] is True
    assert isinstance(result["critical_load_factor"], float)
    assert len(result["buckling_mode"]) == len(frame.nodes)
    for m in result["buckling_mode"]:
        assert set(m) == {"node_id", "ux", "uy", "theta"}
        for key in ("ux", "uy", "theta"):
            assert math.isfinite(m[key])


# ---------------------------------------------------------------------------
# 6) 非法输入 / 机构：与静力入口同一套结构化错误
# ---------------------------------------------------------------------------


def test_mechanism_singular_on_buckling():
    """结构本身是机构（柱底全部松开）时，第一段静力即失败，报 SINGULAR 类错误。"""
    frame, _ = make_axial_column(n_segments=2, boundary="pinned-pinned")
    frame.nodes[0].restraints = [False, False, False]
    # 顶端那点水平约束也不够消除刚体模态，validation 先报支承不足或 solver 报奇异
    with pytest.raises(Exception) as exc:  # noqa: PT011 - 两类均为结构化 FrameError
        analyze_buckling(frame)
    assert exc.value.code in ("INSUFFICIENT_SUPPORT", "SINGULAR_MATRIX")


def test_invalid_geometry_rejected_before_eigensolve():
    frame, _ = make_axial_column(n_segments=2, boundary="pinned-pinned")
    frame.members[0].inertia = -1.0  # 绕过 pydantic 直接破坏模型
    from framesolver.errors import FrameError
    with pytest.raises(FrameError) as exc:
        analyze_buckling(frame)
    assert exc.value.code == "INVALID_PROPERTY"

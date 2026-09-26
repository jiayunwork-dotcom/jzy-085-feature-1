"""弹性稳定（屈曲）分析测试。

覆盖：
- 单元几何刚度的数值、符号方向（纯拉加固 / 纯压削弱抗侧刚度）与零空间；
- 广义特征值求解器的手算可核对算例与非物理特征值甄别；
- 欧拉柱闭式解命中（两端铰支 / 一端固定一端自由 / 一端固定一端铰支），
  以及误差随网格加密的收敛性；
- 荷载整体缩放 -> 临界因子反比例变化；
- 受拉主导 / 无轴力结构报「无正临界因子」而非乱给数；
- 屈曲模态形状与解析模态（正弦半波）对照；
- 两段式数据衔接：几何刚度所用的轴力与静力核算输出的轴力逐一相等；
- 门式刚架（一柱受拉一柱受压、双柱受压两种工况）临界因子的独立复核：
  在 λ_cr 两侧，K_e + λK_g 的最小特征值必须由正转负。

所有容差均为具体数值；欧拉柱取 n = 8 网格，离散误差实测 ~3e-5，
断言容差 5e-4（千分之零点五），留有 15 倍余量。
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_beam, make_euler_column, make_portal_frame
from framesolver.analysis import analyze_frame, run_linear_analysis
from framesolver.constraints import condense_matrix
from framesolver.eigen import lowest_positive_factor
from framesolver.element import local_geometric_stiffness, local_stiffness
from framesolver.errors import NoBucklingModeError
from framesolver.stability import analyze_stability, assemble_geometric_stiffness

# 与闭式解比较的相对容差（千分之零点五；n=8 时实测离散误差 ~3e-5）
TOL_EULER = 5.0e-4
# 荷载缩放反比例关系的相对容差（理论上严格线性，仅含浮点误差）
TOL_SCALING = 1.0e-9


def _euler_closed_form(bc: str, e: float, i_: float, l: float) -> float:
    """欧拉临界力闭式解：P_cr = π²EI / (计算长度)²。"""
    if bc == "pinned":  # 两端铰支，计算长度 = L
        return np.pi**2 * e * i_ / l**2
    if bc == "cantilever":  # 一端固定一端自由，计算长度 = 2L
        return np.pi**2 * e * i_ / (2.0 * l) ** 2
    if bc == "propped":  # 一端固定一端铰支，tan μL = μL 的最小正根
        mu = 4.49340945790906
        return mu**2 * e * i_ / l**2
    raise ValueError(bc)


def _critical_load(frame, p: float) -> float:
    """临界荷载 = 临界荷载因子 × 当前施加的轴压。"""
    return analyze_stability(frame)["critical_load_factor"] * p


# ---------------------------------------------------------------------------
# 单元几何刚度：数值、符号方向、零空间
# ---------------------------------------------------------------------------


def test_geometric_stiffness_entries_and_tension_compression_signs():
    """几何刚度的具体系数，以及纯拉 / 纯压下对抗侧刚度的加减号方向。"""
    l, n_tension = 4.0, 1000.0
    kg_t = local_geometric_stiffness(l, +n_tension)  # 纯拉
    kg_c = local_geometric_stiffness(l, -n_tension)  # 纯压

    # 对称性、关于轴力的奇性（线性）：K_g(−N) = −K_g(+N)
    assert np.allclose(kg_t, kg_t.T, atol=1e-12)
    assert np.allclose(kg_t + kg_c, 0.0, atol=1e-12)

    # 具体系数：f = N/(30L)，kg[1,1] = 36f，kg[2,2] = 4L²f，kg[2,5] = −L²f
    f = n_tension / (30.0 * l)
    assert kg_t[1, 1] == pytest.approx(36.0 * f, rel=1e-12)
    assert kg_t[2, 2] == pytest.approx(4.0 * l**2 * f, rel=1e-12)
    assert kg_t[2, 5] == pytest.approx(-l**2 * f, rel=1e-12)
    assert kg_t[1, 2] == pytest.approx(3.0 * l * f, rel=1e-12)
    # 轴向自由度不参与几何刚度
    assert np.allclose(kg_t[0, :], 0.0) and np.allclose(kg_t[3, :], 0.0)

    # 受拉半正定（加固）、受压半负定（削弱）
    assert np.min(np.linalg.eigvalsh(kg_t)) >= -1e-9 * n_tension
    assert np.max(np.linalg.eigvalsh(kg_c)) <= +1e-9 * n_tension

    # 弯曲型变形上，受拉严格增加横向应变能、受压严格减少
    for d in (
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),   # 两端同号转角（S 形弯曲）
        np.array([0.0, 1.0, 0.0, 0.0, -1.0, 0.0]),  # 两端反向横移（反对称弯曲）
    ):
        assert d @ kg_t @ d > 0.0
        assert d @ kg_c @ d < 0.0
        # 与弹性刚度合成后的加减号方向：受拉更刚、受压更柔
        k_e = local_stiffness(l, 2.1e8, 2.0e-2, 8.0e-5)
        assert d @ (k_e + kg_t) @ d > d @ k_e @ d > d @ (k_e + kg_c) @ d

    # 具体数值锚点：d1ᵀ K_g d1 = NL/5，d1ᵀ K_e d1 = 12EI/L
    d1 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    assert d1 @ kg_t @ d1 == pytest.approx(n_tension * l / 5.0, rel=1e-12)
    k_e = local_stiffness(l, 2.1e8, 2.0e-2, 8.0e-5)
    assert d1 @ k_e @ d1 == pytest.approx(12.0 * 2.1e8 * 8.0e-5 / l, rel=1e-12)


def test_geometric_stiffness_nullspace():
    """刚体平移（轴向、横向）是几何刚度的精确零能模态（dv/dx ≡ 0）。"""
    kg = local_geometric_stiffness(4.0, 1000.0)
    for rigid in (
        np.array([1.0, 0, 0, 1, 0, 0]),  # 轴向平移
        np.array([0.0, 1, 0, 0, 1, 0]),  # 横向平移
    ):
        assert np.allclose(kg @ rigid, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 广义特征值求解器：手算可核对的算例
# ---------------------------------------------------------------------------


def test_eigen_solver_hand_computable_case():
    """K_e = [[2,1],[1,2]]，K_g = −I：K_e φ = λφ，λ_min = 1，φ ∝ (1,−1)。"""
    k_e = np.array([[2.0, 1.0], [1.0, 2.0]])
    k_g = -np.eye(2)
    factor, mode = lowest_positive_factor(k_e, k_g)
    assert factor == pytest.approx(1.0, rel=1e-12)
    # 模态方向 (1, −1)/√2（符号任意）
    unit = mode / np.linalg.norm(mode)
    assert abs(unit[0]) == pytest.approx(1.0 / np.sqrt(2.0), rel=1e-12)
    assert unit[0] * unit[1] < 0.0


def test_eigen_solver_picks_lowest_positive_factor():
    """M = −K_g = diag(1, 0.5)：特征值 λ = 1 与 2，必须取最低的 1。"""
    k_e = np.eye(2)
    k_g = -np.diag([1.0, 0.5])
    factor, mode = lowest_positive_factor(k_e, k_g)
    assert factor == pytest.approx(1.0, rel=1e-12)
    assert abs(mode[0]) > 1e6 * abs(mode[1])  # 模态沿第一个自由度


def test_eigen_solver_discards_nonpositive_eigenvalues():
    """受拉主导（K_g 正定 -> −K_g 负定）与零几何刚度都必须判为无正因子。"""
    k_e = np.array([[2.0, 1.0], [1.0, 2.0]])
    assert lowest_positive_factor(k_e, +np.eye(2)) == (None, None)  # 纯拉
    assert lowest_positive_factor(k_e, np.zeros((2, 2))) == (None, None)  # 无轴力
    assert lowest_positive_factor(np.zeros((0, 0)), np.zeros((0, 0))) == (None, None)


# ---------------------------------------------------------------------------
# 欧拉柱：闭式解命中与网格收敛
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bc", ["pinned", "cantilever", "propped"])
def test_euler_column_matches_closed_form(bc):
    """临界因子 × 施加轴压 应在 5e-4 相对误差内命中欧拉闭式临界力。"""
    frame, p = make_euler_column(n=8, bc=bc)
    exact = _euler_closed_form(bc, p["e"], p["i"], p["l"])
    computed = _critical_load(frame, p["p"])
    assert computed == pytest.approx(exact, rel=TOL_EULER)


def test_euler_error_converges_with_mesh_refinement():
    """分段变细时误差必须单调下降，且细网格误差进入 1e-5 量级。"""
    errors = []
    for n in (1, 2, 4, 8, 16):
        frame, p = make_euler_column(n=n, bc="pinned")
        exact = _euler_closed_form("pinned", p["e"], p["i"], p["l"])
        errors.append(abs(_critical_load(frame, p["p"]) - exact) / exact)
    # 单调收敛
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < 0.2 * coarse
    # n=16 的离散误差（实测 ~2e-6）
    assert errors[-1] < 1.0e-5


def test_cantilever_error_converges_with_mesh_refinement():
    """悬臂柱（计算长度 2L）同样收敛到 π²EI/(2L)²。"""
    errors = []
    for n in (1, 2, 4, 8):
        frame, p = make_euler_column(n=n, bc="cantilever")
        exact = _euler_closed_form("cantilever", p["e"], p["i"], p["l"])
        errors.append(abs(_critical_load(frame, p["p"]) - exact) / exact)
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < 0.2 * coarse
    assert errors[-1] < 1.0e-4


# ---------------------------------------------------------------------------
# 荷载缩放：临界的是荷载总量，因子与荷载成反比
# ---------------------------------------------------------------------------


def test_critical_factor_inversely_proportional_to_load():
    """荷载整体放大 k 倍，临界因子缩小为 1/k（临界荷载总量不变）。"""
    frame1, p1 = make_euler_column(n=8, bc="pinned", p=100.0)
    frame2, p2 = make_euler_column(n=8, bc="pinned", p=200.0)
    frame3, p3 = make_euler_column(n=8, bc="pinned", p=50.0)
    lam1 = analyze_stability(frame1)["critical_load_factor"]
    lam2 = analyze_stability(frame2)["critical_load_factor"]
    lam3 = analyze_stability(frame3)["critical_load_factor"]
    assert lam2 == pytest.approx(lam1 / 2.0, rel=TOL_SCALING)
    assert lam3 == pytest.approx(2.0 * lam1, rel=TOL_SCALING)
    # 临界荷载总量与施加荷载水平无关
    assert lam2 * p2["p"] == pytest.approx(lam1 * p1["p"], rel=TOL_SCALING)


# ---------------------------------------------------------------------------
# 受拉主导 / 无轴力：明确报「无正临界因子」
# ---------------------------------------------------------------------------


def test_tension_column_reports_no_buckling():
    """同一根柱改受拉：不存在正临界因子，抛 NO_BUCKLING_MODE。"""
    frame, _ = make_euler_column(n=8, bc="pinned", tension=True)
    with pytest.raises(NoBucklingModeError) as exc:
        analyze_stability(frame)
    assert exc.value.code == "NO_BUCKLING_MODE"
    assert "屈曲" in exc.value.message


def test_pure_transverse_load_beam_reports_no_buckling():
    """只受横向均布荷载的简支梁：各杆轴力恒为零，不发生弹性屈曲。"""
    frame, _ = make_beam(q_down=3.0, fixed=False)
    with pytest.raises(NoBucklingModeError):
        analyze_stability(frame)


# ---------------------------------------------------------------------------
# 屈曲模态形状
# ---------------------------------------------------------------------------


def test_pinned_column_mode_shape_is_sine_half_wave():
    """两端铰支柱的屈曲模态：uy 沿跨长为正弦半波 sin(πx/L)，ux ≈ 0。"""
    frame, p = make_euler_column(n=8, bc="pinned")
    result = analyze_stability(frame)
    assert result["has_buckling_mode"] is True
    xs = np.array([k * p["l"] / p["n"] for k in range(p["n"] + 1)])
    uy = np.array([m["uy"] for m in result["mode_shape"]])
    ux = np.array([m["ux"] for m in result["mode_shape"]])
    theta = np.array([m["theta"] for m in result["mode_shape"]])
    # 归一化约定：绝对值最大的分量为 +1（此处即跨中 uy）
    assert np.max(np.abs(uy)) == pytest.approx(1.0, rel=1e-12)
    # 模态形状对照解析解（节点值，实测偏差 ~4e-15 / ~3e-7）
    assert np.max(np.abs(uy - np.sin(np.pi * xs / p["l"]))) < 1.0e-6
    assert np.max(np.abs(ux)) < 1.0e-9
    assert np.max(np.abs(theta - (np.pi / p["l"]) * np.cos(np.pi * xs / p["l"]))) < 1.0e-5
    # 支座处被约束自由度恒为零
    assert result["mode_shape"][0]["ux"] == 0.0
    assert result["mode_shape"][0]["uy"] == 0.0
    assert result["mode_shape"][-1]["uy"] == 0.0


def test_eigenpair_satisfies_generalized_equation():
    """特征对必须满足 (K_e + λK_g)φ ≈ 0（相对残差 < 1e-8）。"""
    frame, _ = make_euler_column(n=4, bc="pinned")
    ctx = run_linear_analysis(frame)
    k_g = assemble_geometric_stiffness(
        ctx["ndof"], frame.members, ctx["geometries"], ctx["node_index"],
        ctx["axial_forces"],
    )
    k_e_ff = condense_matrix(ctx["stiffness"], ctx["free_dofs"])
    k_g_ff = condense_matrix(k_g, ctx["free_dofs"])
    factor, mode = lowest_positive_factor(k_e_ff, k_g_ff)
    assert factor is not None
    residual = k_e_ff @ mode + factor * (k_g_ff @ mode)
    scale = float(np.max(np.abs(k_e_ff @ mode)))
    assert float(np.max(np.abs(residual))) < 1.0e-8 * scale


# ---------------------------------------------------------------------------
# 两段式数据衔接：几何刚度用的轴力 == 静力核算输出的轴力
# ---------------------------------------------------------------------------


def test_stability_stage_uses_exact_static_axial_forces():
    """门式刚架水平荷载下：一柱受拉一柱受压，稳定段轴力与静力输出逐一相等。"""
    frame, p = make_portal_frame()
    static_result = analyze_frame(frame)
    ctx = run_linear_analysis(frame)
    for mf in static_result["member_forces"]:
        assert ctx["axial_forces"][mf["member_id"]] == mf["axial_force"]
    # 符号符合力学直觉：左柱受拉、右柱受压（±3P/7），横梁传递 −P/2 压力
    # （A 有限而非无穷，与忽略轴向变形的理论值偏差 ~1e-7，容差取 1e-6）
    assert ctx["axial_forces"]["c1"] == pytest.approx(+3.0 * p["p"] / 7.0, rel=1e-6)
    assert ctx["axial_forces"]["c2"] == pytest.approx(-3.0 * p["p"] / 7.0, rel=1e-6)
    assert ctx["axial_forces"]["b"] == pytest.approx(-0.5 * p["p"], rel=1e-6)


# ---------------------------------------------------------------------------
# 门式刚架整体屈曲：临界因子的独立复核（切线刚度在 λ_cr 处由正定转奇异）
# ---------------------------------------------------------------------------


def _tangent_stiffness_min_eigenvalue(frame, factor: float) -> float:
    """K_e + λK_g 缩聚后的最小特征值（独立复核用，不经过特征值求解器）。"""
    ctx = run_linear_analysis(frame)
    k_g = assemble_geometric_stiffness(
        ctx["ndof"], frame.members, ctx["geometries"], ctx["node_index"],
        ctx["axial_forces"],
    )
    k_e_ff = condense_matrix(ctx["stiffness"], ctx["free_dofs"])
    k_g_ff = condense_matrix(k_g, ctx["free_dofs"])
    tangent = k_e_ff + factor * k_g_ff
    return float(np.linalg.eigvalsh(0.5 * (tangent + tangent.T))[0])


def test_portal_frame_lateral_load_buckling_factor_bracketed():
    """水平荷载门式刚架（一柱拉一柱压）：λ_cr 两侧切线刚度必须由正转负。"""
    frame, _ = make_portal_frame()
    result = analyze_stability(frame)
    lam = result["critical_load_factor"]
    assert np.isfinite(lam) and lam > 0.0
    assert _tangent_stiffness_min_eigenvalue(frame, 0.99 * lam) > 0.0
    assert _tangent_stiffness_min_eigenvalue(frame, 1.01 * lam) < 0.0


def test_portal_frame_vertical_load_buckling_factor_bracketed():
    """柱顶竖向力门式刚架（双柱受压）：λ_cr 两侧切线刚度同样由正转负。"""
    frame, p = make_portal_frame()
    frame.nodal_loads[0].fx = 0.0
    frame.nodal_loads[0].fy = -p["p"]
    frame.nodal_loads.append(type(frame.nodal_loads[0])(node_id="3", fy=-p["p"]))
    result = analyze_stability(frame)
    lam = result["critical_load_factor"]
    assert np.isfinite(lam) and lam > 0.0
    assert _tangent_stiffness_min_eigenvalue(frame, 0.99 * lam) > 0.0
    assert _tangent_stiffness_min_eigenvalue(frame, 1.01 * lam) < 0.0

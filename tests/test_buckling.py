"""弹性稳定（屈曲）分析测试。

覆盖四条主线：

1. **经典解校核**：等截面直杆在轴压下的弹性临界荷载有欧拉闭式解
   P_cr = π²EI / (μL)²（μ 由两端约束决定：两端铰接 μ=1，一端固定一端
   自由 μ=2，一端固定一端铰接 μ≈0.699）。服务输出的临界荷载因子乘上
   施加的轴压，须在千分之一相对误差内命中闭式解，且误差随分段变细
   单调收敛（一致几何刚度下约为 4 阶收敛）。
2. **几何刚度符号方向**：轴力为正（拉）加固抗侧 / 抗弯刚度，为负
   （压）削弱；单元级用具体数值守死。
3. **非物理特征值甄别**：受拉主导结构、零荷载结构必须报
   NO_BUCKLING_MODE，绝不返回硬凑的数字；荷载整体缩放 k 倍，
   临界因子须精确变为 1/k（临界的是荷载总量）。
4. **兼容性**：/solve 静力入口的响应结构与数值在本次改动后不变。

容差全部写成具体数值。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from framesolver.analysis import (
    _assemble_validated_system,
    _member_axial_forces,
    _solve_first_order,
    analyze_buckling,
    analyze_frame,
)
from framesolver.assembly import assemble_matrix
from framesolver.element import local_stiffness
from framesolver.errors import NoBucklingModeError, SingularMatrixError
from framesolver.geometry import stiffness_to_global, transformation_matrix
from framesolver.main import app
from framesolver.models import FrameInput, Member, Node, NodalLoad
from framesolver.stability import (
    local_geometric_stiffness,
    normalize_mode_shape,
    solve_critical_load,
)

client = TestClient(app)

# ---------------------------------------------------------------------------
# 容差（具体数值）
# ---------------------------------------------------------------------------
TOL_EULER_REL = 1.0e-3      # 临界力与欧拉闭式解的相对误差上限（千分之一）
TOL_CONVERGED_REL = 1.0e-5  # 细分到 16 段后的相对误差上限
TOL_SCALING_REL = 1.0e-9    # 荷载缩放后临界因子反比例变化的相对误差上限
TOL_MODE_SHAPE = 1.0e-6     # 屈曲模态与正弦半波的逐点偏差上限
TOL_RESIDUAL_REL = 1.0e-8   # (K_e + λK_g)φ = 0 的相对残差上限
TOL_EXACT = 1.0e-12         # 单元级代数恒等式

# 统一的柱截面参数（kN, m）
E, A, I, L, P = 2.1e8, 2.0e-2, 8.0e-5, 4.0, 10.0
EI = E * I
# 固定-铰接柱的精确有效长度因子：tan(x)=x 的最小正根
KL_FIXED_PINNED = 4.493409457909064


def make_column(n_seg: int, top: list[bool], bottom: list[bool], load: float = -P):
    """竖直等截面柱：底端 y=0、顶端 y=L，n_seg 个单元，顶端施加竖向力。

    加载端的轴向自由度必须放开（否则轴压直接流进支座、杆件不受力）；
    横向约束方式决定计算长度。
    """
    nodes = []
    for k in range(n_seg + 1):
        r = [False, False, False]
        if k == 0:
            r = bottom
        elif k == n_seg:
            r = top
        nodes.append(Node(id=str(k), x=0.0, y=L * k / n_seg, restraints=list(r)))
    members = [
        Member(id=f"m{k}", node_i=str(k), node_j=str(k + 1),
               elastic_modulus=E, area=A, inertia=I)
        for k in range(n_seg)
    ]
    return FrameInput(
        nodes=nodes,
        members=members,
        nodal_loads=[NodalLoad(node_id=str(n_seg), fx=0.0, fy=load, moment=0.0)],
    )


def make_pin_pin(n_seg: int = 8, load: float = -P):
    """两端铰接柱：两端横向锁死、转角自由，顶端轴向放开以承受轴压。"""
    return make_column(n_seg, top=[True, False, False], bottom=[True, True, False], load=load)


def make_cantilever(n_seg: int = 8, load: float = -P):
    """一端固定一端自由柱（悬臂压杆）。"""
    return make_column(n_seg, top=[False, False, False], bottom=[True, True, True], load=load)


def make_fixed_pinned(n_seg: int = 16, load: float = -P):
    """一端固定一端铰接柱。"""
    return make_column(n_seg, top=[True, False, False], bottom=[True, True, True], load=load)


# ---------------------------------------------------------------------------
# 单元几何刚度：符号方向与代数性质
# ---------------------------------------------------------------------------

def test_geometric_stiffness_symmetric_and_no_axial_entries():
    kg = local_geometric_stiffness(3.0, 150.0)
    assert np.allclose(kg, kg.T, atol=TOL_EXACT)
    # 轴向自由度（0 和 3）对应的行、列必须全为零：
    # 几何刚度只削弱 / 加固横向与转动，不影响轴向刚度
    assert np.allclose(kg[0, :], 0.0, atol=TOL_EXACT)
    assert np.allclose(kg[3, :], 0.0, atol=TOL_EXACT)
    assert np.allclose(kg[:, 0], 0.0, atol=TOL_EXACT)
    assert np.allclose(kg[:, 3], 0.0, atol=TOL_EXACT)


def test_geometric_stiffness_linear_in_axial_force():
    kg_n = local_geometric_stiffness(4.0, 100.0)
    kg_2n = local_geometric_stiffness(4.0, 200.0)
    kg_neg = local_geometric_stiffness(4.0, -100.0)
    assert np.allclose(kg_2n, 2.0 * kg_n, atol=TOL_EXACT)
    # 拉压反号：同幅值拉力与压力的几何刚度严格互为相反数
    assert np.allclose(kg_neg, -kg_n, atol=TOL_EXACT)


def test_geometric_stiffness_rigid_translation_zero_energy():
    """刚体横移不产生几何刚度能量（v′ = 0）。"""
    kg = local_geometric_stiffness(4.0, -100.0)
    rigid_translation = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    assert np.allclose(kg @ rigid_translation, 0.0, atol=TOL_EXACT)


def test_geometric_stiffness_sign_tension_stiffens_compression_softens():
    """符号方向的核心测试：纯曲率变形模态下的几何刚度能量。

    取 v = [0,0,1,0,0,-1]（两端转角反向、无横移的纯弯曲模态），
    由 Hermite 形函数积分得 vᵀK_g v = N·L/3：
    受拉（N>0）为正（加固），受压（N<0）为负（削弱）。
    """
    n, l = 120.0, 3.0
    curvature = np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0])
    e_tension = curvature @ local_geometric_stiffness(l, n) @ curvature
    e_compression = curvature @ local_geometric_stiffness(l, -n) @ curvature
    assert e_tension == pytest.approx(n * l / 3.0, rel=TOL_EXACT)
    assert e_tension > 0.0
    assert e_compression == pytest.approx(-n * l / 3.0, rel=TOL_EXACT)
    assert e_compression < 0.0


def test_geometric_stiffness_lateral_stiffness_shift_exact_values():
    """元素级「加固 / 削弱」定量测试：锁定横移、放开转角（抗侧转动刚度）。

    L=4, EI=16800, N=±100 时，缩聚到转角自由度的 2×2 切线刚度
    K_rot = K_e_rot + K_g_rot 的特征值：
      弹性      : [8400, 25200]
      受拉 +100 : [8466.666…, 25240]   （每个特征值都被抬高）
      受压 −100 : [8333.333…, 25160]   （每个特征值都被压低）
    """
    l, ei, n = 4.0, 16800.0, 100.0
    ke = local_stiffness(l, 2.1e8, 1.0, ei / 2.1e8)
    rot = [2, 5]  # 两端转角自由度
    ke_rot = ke[np.ix_(rot, rot)]
    kg_t = local_geometric_stiffness(l, n)[np.ix_(rot, rot)]
    kg_c = local_geometric_stiffness(l, -n)[np.ix_(rot, rot)]

    eig_elastic = np.linalg.eigvalsh(ke_rot)
    eig_tension = np.linalg.eigvalsh(ke_rot + kg_t)
    eig_compression = np.linalg.eigvalsh(ke_rot + kg_c)

    assert eig_elastic == pytest.approx([8400.0, 25200.0], rel=TOL_EXACT)
    assert eig_tension == pytest.approx([8400.0 + 200.0 / 3.0, 25240.0], rel=TOL_EXACT)
    assert eig_compression == pytest.approx([8400.0 - 200.0 / 3.0, 25160.0], rel=TOL_EXACT)
    # 方向断言：受拉逐个抬高、受压逐个压低
    assert np.all(eig_tension > eig_elastic)
    assert np.all(eig_compression < eig_elastic)


# ---------------------------------------------------------------------------
# 广义特征值求解：过滤非物理特征值
# ---------------------------------------------------------------------------

def test_solve_critical_load_simple_diagonal():
    """K_e = I，K_g = −diag(2,1)：临界方程 (1−2λ)(1−λ)=0，λ_cr = 1/2。"""
    k = np.eye(2)
    kg = -np.diag([2.0, 1.0])
    lam, phi = solve_critical_load(k, kg)
    assert lam == pytest.approx(0.5, rel=TOL_EXACT)
    # 模态沿第一个坐标方向
    assert abs(abs(phi[0]) - 1.0) < 1e-9 or abs(phi[0]) > abs(phi[1])
    assert abs(phi[1]) < 1e-9


def test_solve_critical_load_tension_only_raises():
    """K_g 正定（全受拉）：不存在正临界因子。"""
    k = np.eye(2)
    kg = np.diag([2.0, 1.0])
    with pytest.raises(NoBucklingModeError) as exc:
        solve_critical_load(k, kg)
    assert exc.value.code == "NO_BUCKLING_MODE"
    assert exc.value.http_status == 422
    assert "不发生弹性屈曲" in exc.value.message


def test_solve_critical_load_zero_geometric_stiffness_raises():
    """零荷载 -> K_g ≡ 0：特征值问题退化，报无屈曲而非返回无穷大。"""
    with pytest.raises(NoBucklingModeError):
        solve_critical_load(np.eye(3), np.zeros((3, 3)))


def test_solve_critical_load_non_spd_elastic_raises():
    """弹性刚度非正定（机构残留）必须按奇异报错，不得硬解。"""
    k = np.diag([1.0, -1.0])
    kg = -np.eye(2)
    with pytest.raises(SingularMatrixError):
        solve_critical_load(k, kg)


def test_normalize_mode_shape_convention():
    phi = normalize_mode_shape(np.array([0.5, -2.0, 0.25]))
    assert np.max(np.abs(phi)) == pytest.approx(1.0)
    # 最大绝对分量取正号（符号确定、可复现）
    assert phi[1] == pytest.approx(1.0)
    assert phi[0] == pytest.approx(-0.25)
    # 零向量原样返回，不产生 NaN
    assert np.allclose(normalize_mode_shape(np.zeros(3)), 0.0)


# ---------------------------------------------------------------------------
# 欧拉闭式解校核（积分级）
# ---------------------------------------------------------------------------

def test_euler_pin_pin_column():
    """两端铰接：P_cr = π²EI/L²，8 段离散误差须 < 0.1%（实际约 3e-5）。"""
    result = analyze_buckling(make_pin_pin(n_seg=8))
    assert result["success"] is True
    assert result["has_buckling_mode"] is True
    p_cr = result["critical_load_factor"] * P
    p_euler = math.pi**2 * EI / L**2
    assert p_cr == pytest.approx(p_euler, rel=TOL_EULER_REL)


def test_euler_cantilever_column():
    """一端固定一端自由：P_cr = π²EI/(2L)²，8 段离散误差须 < 0.1%。"""
    result = analyze_buckling(make_cantilever(n_seg=8))
    p_cr = result["critical_load_factor"] * P
    p_euler = math.pi**2 * EI / (2.0 * L) ** 2
    assert p_cr == pytest.approx(p_euler, rel=TOL_EULER_REL)


def test_euler_fixed_pinned_column():
    """一端固定一端铰接：P_cr = (4.4934…)²EI/L²，16 段误差须 < 0.1%。"""
    result = analyze_buckling(make_fixed_pinned(n_seg=16))
    p_cr = result["critical_load_factor"] * P
    p_euler = KL_FIXED_PINNED**2 * EI / L**2
    assert p_cr == pytest.approx(p_euler, rel=TOL_EULER_REL)


def test_euler_error_converges_with_refinement():
    """误差随分段变细单调收敛：n = 2,4,8,16 误差严格递减，
    且 16 段时 < 1e-5（一致几何刚度的收敛阶约 4 阶，每加密一倍
    误差约缩小 16 倍，这里只要求单调 + 终值达标）。"""
    p_euler = math.pi**2 * EI / L**2
    errors = []
    for n_seg in (2, 4, 8, 16):
        result = analyze_buckling(make_pin_pin(n_seg=n_seg))
        p_cr = result["critical_load_factor"] * P
        errors.append(abs(p_cr - p_euler) / p_euler)
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < coarse, f"误差未随加密单调下降：{errors}"
    assert errors[0] < 1.0e-2
    assert errors[-1] < TOL_CONVERGED_REL


def test_buckling_mode_shape_is_half_sine():
    """两端铰接柱的屈曲模态是半波正弦 sin(πy/L)（横向分量逐点核对）。"""
    n_seg = 8
    result = analyze_buckling(make_pin_pin(n_seg=n_seg))
    mode = {m["node_id"]: m for m in result["buckling_mode"]}
    # 归一化约定：最大绝对分量 = 1，且该分量为正
    all_components = [
        m[key] for m in result["buckling_mode"] for key in ("ux", "uy", "theta")
    ]
    assert max(abs(v) for v in all_components) == pytest.approx(1.0)
    # 柱竖直放置，屈曲为横向（x 向）半波正弦
    sign = 1.0 if mode[str(n_seg // 2)]["ux"] >= 0 else -1.0
    for k in range(n_seg + 1):
        y = L * k / n_seg
        assert sign * mode[str(k)]["ux"] == pytest.approx(
            math.sin(math.pi * y / L), abs=TOL_MODE_SHAPE
        )
    # 确定性：同一输入两次求解，模态逐分量完全一致
    again = analyze_buckling(make_pin_pin(n_seg=n_seg))
    for a, b in zip(result["buckling_mode"], again["buckling_mode"]):
        assert (a["ux"], a["uy"], a["theta"]) == (b["ux"], b["uy"], b["theta"])


# ---------------------------------------------------------------------------
# 荷载缩放：临界的是荷载总量
# ---------------------------------------------------------------------------

def test_load_scaling_inverts_critical_factor():
    """荷载整体放大 k 倍，临界因子精确变为 1/k（K_g 随轴力线性变化）。"""
    lam_1 = analyze_buckling(make_pin_pin(n_seg=8, load=-P))["critical_load_factor"]
    lam_2 = analyze_buckling(make_pin_pin(n_seg=8, load=-2.0 * P))["critical_load_factor"]
    lam_half = analyze_buckling(make_pin_pin(n_seg=8, load=-0.5 * P))["critical_load_factor"]
    assert lam_2 == pytest.approx(lam_1 / 2.0, rel=TOL_SCALING_REL)
    assert lam_half == pytest.approx(2.0 * lam_1, rel=TOL_SCALING_REL)


# ---------------------------------------------------------------------------
# 非物理情形：受拉主导 / 零荷载必须报「无正临界因子」
# ---------------------------------------------------------------------------

def test_tension_dominated_column_reports_no_buckling():
    """同一张柱模型、荷载反向（受拉）：必须报 NO_BUCKLING_MODE 而非给数。"""
    with pytest.raises(NoBucklingModeError) as exc:
        analyze_buckling(make_pin_pin(n_seg=8, load=+P))
    assert exc.value.code == "NO_BUCKLING_MODE"
    assert "不发生弹性屈曲" in exc.value.message


def test_zero_load_reports_no_buckling():
    """零荷载不产生轴力，几何刚度为零：报 NO_BUCKLING_MODE。"""
    with pytest.raises(NoBucklingModeError):
        analyze_buckling(make_pin_pin(n_seg=8, load=0.0))


def test_fully_restrained_structure_reports_no_buckling():
    """全部自由度锁死：没有可失稳的变形模态，报 NO_BUCKLING_MODE 而非崩溃。"""
    frame = FrameInput(
        nodes=[
            Node(id="1", x=0.0, y=0.0, restraints=[True, True, True]),
            Node(id="2", x=0.0, y=L, restraints=[True, True, True]),
        ],
        members=[Member(id="m", node_i="1", node_j="2",
                        elastic_modulus=E, area=A, inertia=I)],
        nodal_loads=[NodalLoad(node_id="2", fx=0.0, fy=-P, moment=0.0)],
    )
    with pytest.raises(NoBucklingModeError):
        analyze_buckling(frame)


# ---------------------------------------------------------------------------
# 刚架层级：屈曲方程残差恒等式（含分布荷载路径）
# ---------------------------------------------------------------------------

def _buckling_residual(frame):
    """返回 (K_e + λK_g)φ 在自由自由度上的相对残差（白盒核对恒等式）。"""
    system = _assemble_validated_system(frame)
    displacement = _solve_first_order(system)
    axial = _member_axial_forces(system, displacement)
    kg_list = []
    for member, geom, k_local, t, eq_local in system["local_data"]:
        kg_local = local_geometric_stiffness(geom.length, axial[member.id])
        kg_list.append(stiffness_to_global(kg_local, t))
    kg = assemble_matrix(system["ndof"], frame.members, system["node_index"], kg_list)

    result = analyze_buckling(frame)
    lam = result["critical_load_factor"]
    phi = np.zeros(system["ndof"])
    for m in result["buckling_mode"]:
        idx = system["node_index"][m["node_id"]]
        phi[3 * idx: 3 * idx + 3] = [m["ux"], m["uy"], m["theta"]]

    residual = (system["stiffness"] + lam * kg) @ phi
    free = system["free_dofs"]
    scale = float(np.max(np.abs(system["stiffness"] @ phi)))
    return float(np.max(np.abs(residual[free]))) / scale, result


def test_portal_frame_vertical_load_buckling_residual():
    """门式刚架两柱顶受竖向压力：λ_cr 为正有限值，模态为侧移型，
    且 (K_e + λK_g)φ = 0 在自由自由度上满足到 1e-8 相对残差。"""
    frame = FrameInput(
        nodes=[
            Node(id="1", x=0.0, y=0.0, restraints=[True, True, True]),
            Node(id="2", x=0.0, y=L, restraints=[False, False, False]),
            Node(id="3", x=L, y=L, restraints=[False, False, False]),
            Node(id="4", x=L, y=0.0, restraints=[True, True, True]),
        ],
        members=[
            Member(id="c1", node_i="1", node_j="2", elastic_modulus=E, area=A, inertia=I),
            Member(id="b", node_i="2", node_j="3", elastic_modulus=E, area=A, inertia=I),
            Member(id="c2", node_i="4", node_j="3", elastic_modulus=E, area=A, inertia=I),
        ],
        nodal_loads=[
            NodalLoad(node_id="2", fx=0.0, fy=-P, moment=0.0),
            NodalLoad(node_id="3", fx=0.0, fy=-P, moment=0.0),
        ],
    )
    rel_residual, result = _buckling_residual(frame)
    assert rel_residual < TOL_RESIDUAL_REL
    assert math.isfinite(result["critical_load_factor"])
    assert result["critical_load_factor"] > 0.0
    # 对称刚架受对称竖向荷载，首阶失稳为侧移模态：两柱顶横移同向同幅
    mode = {m["node_id"]: m for m in result["buckling_mode"]}
    assert mode["2"]["ux"] == pytest.approx(mode["3"]["ux"], rel=1e-9)
    assert abs(mode["2"]["ux"]) == pytest.approx(1.0)  # 归一化后横向分量最大


def test_frame_with_distributed_load_buckling_residual():
    """含杆件均布荷载的刚架：第一段轴力来自真实静力解（含等效节点力），
    屈曲方程残差同样须满足 1e-8。"""
    from conftest import make_mixed_frame

    frame, _ = make_mixed_frame()
    rel_residual, result = _buckling_residual(frame)
    assert rel_residual < TOL_RESIDUAL_REL
    assert result["critical_load_factor"] > 0.0


# ---------------------------------------------------------------------------
# HTTP 入口
# ---------------------------------------------------------------------------

def test_http_buckling_success_response_shape():
    resp = client.post("/solve/buckling", json=make_pin_pin(n_seg=8).model_dump())
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert set(body) == {"success", "critical_load_factor", "has_buckling_mode", "buckling_mode"}
    assert body["has_buckling_mode"] is True
    assert math.isfinite(body["critical_load_factor"])
    assert body["critical_load_factor"] > 0.0
    assert len(body["buckling_mode"]) == 9
    for entry in body["buckling_mode"]:
        assert set(entry) == {"node_id", "ux", "uy", "theta"}
        assert all(math.isfinite(entry[k]) for k in ("ux", "uy", "theta"))
    # 响应中绝不出现 NaN / Infinity
    text = resp.text
    assert "NaN" not in text and "Infinity" not in text


def test_http_buckling_euler_value():
    """HTTP 层级同样命中欧拉解：λ_cr × P ≈ π²EI/L²（0.1% 内）。"""
    resp = client.post("/solve/buckling", json=make_pin_pin(n_seg=8).model_dump())
    body = resp.json()
    p_cr = body["critical_load_factor"] * P
    assert p_cr == pytest.approx(math.pi**2 * EI / L**2, rel=TOL_EULER_REL)


def test_http_buckling_tension_returns_structured_error():
    payload = make_pin_pin(n_seg=8, load=+P).model_dump()  # 受拉
    resp = client.post("/solve/buckling", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == "NO_BUCKLING_MODE"
    assert body["error"]["message"]


def test_http_buckling_invalid_model_shares_static_error_codes():
    """非法模型在稳定入口的报错代码与静力入口完全一致。"""
    payload = make_pin_pin(n_seg=4).model_dump()
    payload["nodes"][1]["id"] = "0"  # 制造重复节点编号
    resp = client.post("/solve/buckling", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "DUPLICATE_NODE_ID"

    resp_bad = client.post("/solve/buckling", content="{not json",
                           headers={"content-type": "application/json"})
    assert resp_bad.status_code == 422
    assert resp_bad.json()["error"]["code"] == "INVALID_REQUEST"


def test_static_solve_endpoint_unchanged():
    """静力入口的响应结构与关键数值不因新增稳定分析而改变。"""
    resp = client.post("/solve", json=make_pin_pin(n_seg=8).model_dump())
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"success", "displacements", "member_forces", "reactions"}
    for d in body["displacements"]:
        assert set(d) == {"node_id", "ux", "uy", "theta"}
    for m in body["member_forces"]:
        assert set(m) == {"member_id", "node_i", "node_j", "end_i", "end_j", "axial_force"}
    for r in body["reactions"]:
        assert set(r) == {"node_id", "fx", "fy", "moment", "restrained"}
    # 轴压柱的静力解：各杆轴力 = -P（压），与稳定分析第一段共用同一结果
    for m in body["member_forces"]:
        assert m["axial_force"] == pytest.approx(-P, rel=1e-9)

    spec = client.get("/openapi.json").json()
    assert "/solve" in spec["paths"]
    assert "/solve/buckling" in spec["paths"]

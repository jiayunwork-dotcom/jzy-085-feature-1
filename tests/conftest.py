"""pytest 公共夹具与辅助函数。"""

from __future__ import annotations

import numpy as np
import pytest

from framesolver.models import (
    DistributedLoad,
    FrameInput,
    Member,
    Node,
    NodalLoad,
)

# ---------------------------------------------------------------------------
# 统一的数值容差（写死具体数值，不做“返回了数字”式断言）
# ---------------------------------------------------------------------------
TOL_FORCE = 1.0e-8     # 力 / 轴力 / 剪力，量纲 kN
TOL_MOMENT = 1.0e-8    # 弯矩，量纲 kN·m
TOL_DISP = 1.0e-10     # 位移残差级
TOL_THEORY = 2.0e-3    # 与忽略轴向变形的教材闭式解之间的相对容差（0.2%）


def make_portal_frame(p: float = 10.0, l: float = 4.0):
    """两柱底固定的单跨门式刚架，水平集中力 P 作用在左柱顶（节点 2），+x 向。"""
    e, a, i_ = 2.1e8, 2.0e4, 8.0e-3  # kN, m 单位制；A 很大 => 轴向变形可忽略
    return FrameInput(
        nodes=[
            Node(id="1", x=0.0, y=0.0, restraints=[True, True, True]),
            Node(id="2", x=0.0, y=l, restraints=[False, False, False]),
            Node(id="3", x=l, y=l, restraints=[False, False, False]),
            Node(id="4", x=l, y=0.0, restraints=[True, True, True]),
        ],
        members=[
            Member(id="c1", node_i="1", node_j="2",
                   elastic_modulus=e, area=a, inertia=i_),
            Member(id="b", node_i="2", node_j="3",
                   elastic_modulus=e, area=a, inertia=i_),
            Member(id="c2", node_i="4", node_j="3",
                   elastic_modulus=e, area=a, inertia=i_),
        ],
        nodal_loads=[NodalLoad(node_id="2", fx=p, fy=0.0, moment=0.0)],
    ), dict(e=e, a=a, i=i_, l=l, p=p)


def make_beam(q_down: float = 3.0, l: float = 4.0, fixed: bool = True):
    """单根梁满跨垂直均布荷载（q 向下给绝对值）；fixed 切换两端固定 / 两端铰支。"""
    e, a, i_ = 2.1e8, 2.0e-2, 8.0e-5
    r = [True, True, True] if fixed else [True, True, False]
    return FrameInput(
        nodes=[
            Node(id="A", x=0.0, y=0.0, restraints=r),
            Node(id="B", x=l, y=0.0, restraints=r),
        ],
        members=[Member(id="beam", node_i="A", node_j="B",
                        elastic_modulus=e, area=a, inertia=i_)],
        distributed_loads=[DistributedLoad(member_id="beam",
                                           load_type="uniform_transverse",
                                           qy=-q_down)],
    ), dict(e=e, a=a, i=i_, l=l, q=q_down)


def make_euler_column(n: int = 8, l: float = 4.0, p: float = 100.0,
                      bc: str = "pinned", tension: bool = False):
    """等截面直柱（水平放置，x ∈ [0, L]），离散为 n 个单元，右端节点受轴向力。

    bc 决定两端约束方式（即欧拉公式中的计算长度）：
    - "pinned"     两端铰支：P_cr = π²EI/L²
    - "cantilever" 左端固定、右端自由：P_cr = π²EI/(2L)²
    - "propped"    左端固定、右端铰支：P_cr = μ²EI/L²，μ = 4.49340945790906
                   （tan μ = μ 的最小正根）

    tension = True 时轴向力改为受拉（用于「无正临界因子」测试）。
    """
    e, a, i_ = 2.1e8, 2.0e-2, 8.0e-5  # kN, m 单位制
    right_restraints = {
        "pinned": [False, True, False],      # 铰支：轴向自由、横向锁死、转角放开
        "cantilever": [False, False, False],  # 自由端
        "propped": [False, True, False],
    }
    nodes = []
    for k in range(n + 1):
        x = l * k / n
        if k == 0:
            restraints = [True, True, bc != "pinned"]  # 铰支放开转角；固定端锁死
        elif k == n:
            restraints = right_restraints[bc]
        else:
            restraints = [False, False, False]
        nodes.append(Node(id=str(k + 1), x=x, y=0.0, restraints=restraints))
    members = [
        Member(id=f"m{k + 1}", node_i=str(k + 1), node_j=str(k + 2),
               elastic_modulus=e, area=a, inertia=i_)
        for k in range(n)
    ]
    direction = 1.0 if tension else -1.0  # 压力把右端推向左端支座（-x）
    frame = FrameInput(
        nodes=nodes,
        members=members,
        nodal_loads=[NodalLoad(node_id=str(n + 1), fx=direction * p)],
    )
    return frame, dict(e=e, a=a, i=i_, l=l, p=p, n=n)


def make_mixed_frame():
    """带斜杆、铰支 / 固定混合支承、节点荷载 + 分布荷载的刚架（供平衡恒等式测试）。"""
    e, a, i_ = 3.0e7, 5.0e-2, 2.0e-4
    frame = FrameInput(
        nodes=[
            Node(id="n1", x=0.0, y=0.0, restraints=[True, True, True]),      # 固定
            Node(id="n2", x=0.0, y=3.0, restraints=[False, False, False]),
            Node(id="n3", x=4.0, y=3.0, restraints=[False, False, False]),
            Node(id="n4", x=4.0, y=0.0, restraints=[True, True, False]),     # 铰支
        ],
        members=[
            Member(id="m12", node_i="n1", node_j="n2",
                   elastic_modulus=e, area=a, inertia=i_),
            Member(id="m23", node_i="n2", node_j="n3",
                   elastic_modulus=e, area=a, inertia=i_),
            Member(id="m43", node_i="n4", node_j="n3",
                   elastic_modulus=e, area=a, inertia=2.0 * i_),
        ],
        nodal_loads=[
            NodalLoad(node_id="n2", fx=12.5, fy=-7.25, moment=3.75),
            NodalLoad(node_id="n3", fx=0.0, fy=-5.0, moment=-2.5),
        ],
        distributed_loads=[
            DistributedLoad(member_id="m23", load_type="uniform_transverse", qy=-4.0),
        ],
    )
    return frame, dict(e=e, a=a, i=i_)


# ---------------------------------------------------------------------------
# 结果索引辅助
# ---------------------------------------------------------------------------

def by_id(items, key="node_id"):
    return {item[key]: item for item in items}


def member_direction(frame, member_id):
    """返回杆件方向 (c, s, length)。"""
    members = {m.id: m for m in frame.members}
    nodes = {n.id: n for n in frame.nodes}
    m = members[member_id]
    ni, nj = nodes[m.node_i], nodes[m.node_j]
    dx, dy = nj.x - ni.x, nj.y - ni.y
    length = float(np.hypot(dx, dy))
    return dx / length, dy / length, length


def section_sagging_moment(frame, result, member_id, x_from_i, qy=0.0):
    """取 i 端到距 i 端 x 处截面的隔离体，求标量弯矩（下侧受拉为正）。

    直接由隔离体外力对截面取矩推导（输出杆端力为“杆对节点”，
    故 i 端节点对杆的力取反号）：

        M_sag(x) = M_i - V_i·x + qy·x²/2

    V_i、M_i 为输出的 i 端局部剪力 / 弯矩，qy 为局部 +ȳ 向荷载。
    已用简支梁（跨中 qL²/8）、固定梁（端 qL²/12、跨中 qL²/24、
    L/4 处 qL²/96）教材解验证。
    """
    m = next(item for item in result["member_forces"] if item["member_id"] == member_id)
    v_i = m["end_i"]["shear"]
    m_i = m["end_i"]["moment"]
    return m_i - v_i * x_from_i + 0.5 * qy * x_from_i ** 2


def member_global_end_forces(frame, result, member_id):
    """把一根杆的六维局部杆端力（杆对节点）转到整体坐标。"""
    nodes = {n.id: n for n in frame.nodes}
    members = {m.id: m for m in frame.members}
    m = members[member_id]
    ni, nj = nodes[m.node_i], nodes[m.node_j]
    dx, dy = nj.x - ni.x, nj.y - ni.y
    length = float(np.hypot(dx, dy))
    c, s = dx / length, dy / length
    r = np.array([[c, s], [-s, c]])
    f = next(item for item in result["member_forces"] if item["member_id"] == member_id)
    fi = r.T @ np.array([f["end_i"]["axial"], f["end_i"]["shear"]])
    fj = r.T @ np.array([f["end_j"]["axial"], f["end_j"]["shear"]])
    return fi, fj


@pytest.fixture
def tol():
    return dict(force=TOL_FORCE, moment=TOL_MOMENT, disp=TOL_DISP, theory=TOL_THEORY)

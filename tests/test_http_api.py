"""HTTP 入口测试（FastAPI TestClient，不起真实端口）。

覆盖：
- 正常求解返回 success=true 与三类结果；
- 各类非法模型返回结构化错误 {success:false,error:{code,message}}，
  且 HTTP 状态码与错误类型对应、不出现 500 / HTML 栈；
- 请求体 JSON 不合 Schema 时返回 INVALID_REQUEST；
- 错误响应中绝不出现 NaN；
- 分布荷载与节点荷载混合的端到端求解。
"""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from conftest import make_euler_column, make_mixed_frame, make_portal_frame
from framesolver.main import app

client = TestClient(app)


def _dump(frame):
    return frame.model_dump()


def test_solve_success_shape():
    frame, _ = make_portal_frame()
    resp = client.post("/solve", json=_dump(frame))
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert {d["node_id"] for d in body["displacements"]} == {"1", "2", "3", "4"}
    for d in body["displacements"]:
        assert set(d) == {"node_id", "ux", "uy", "theta"}
        assert all(math.isfinite(v) for k, v in d.items() if k != "node_id")
    assert len(body["member_forces"]) == 3
    for m in body["member_forces"]:
        for end in ("end_i", "end_j"):
            assert set(m[end]) == {"axial", "shear", "moment"}
    assert len(body["reactions"]) == 4
    # 只有支座节点的反力分量可能非零
    for r in body["reactions"]:
        assert "restrained" in r and len(r["restrained"]) == 3


def test_solve_mixed_loads_end_to_end():
    frame, _ = make_mixed_frame()
    resp = client.post("/solve", json=_dump(frame))
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    # 全部结果有限
    for group in ("displacements", "member_forces", "reactions"):
        text = str(body[group])
        assert "NaN" not in text and "Infinity" not in text


def test_duplicate_node_id_http_error():
    frame, _ = make_portal_frame()
    payload = _dump(frame)
    payload["nodes"][1]["id"] = "1"  # 制造重复编号
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == "DUPLICATE_NODE_ID"
    assert body["error"]["message"]


def test_member_to_ghost_node_http_error():
    frame, _ = make_portal_frame()
    payload = _dump(frame)
    payload["members"][0]["node_j"] = "ghost"
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "MEMBER_NODE_NOT_FOUND"


def test_zero_length_member_http_error():
    frame, _ = make_portal_frame()
    payload = _dump(frame)
    payload["nodes"][1]["y"] = 0.0  # 节点 2 与节点 1 重合
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "ZERO_LENGTH_MEMBER"


@pytest.mark.parametrize("field", ["elastic_modulus", "area", "inertia"])
def test_nonpositive_property_http_error(field):
    frame, _ = make_portal_frame()
    payload = _dump(frame)
    payload["members"][0][field] = -1.0
    resp = client.post("/solve", json=payload)
    # Pydantic 的 gt 校验先拦截 -> INVALID_REQUEST；绕过 Pydantic 时为 INVALID_PROPERTY
    assert resp.status_code in (400, 422)
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] in ("INVALID_REQUEST", "INVALID_PROPERTY")


def test_disconnected_http_error():
    frame, _ = make_portal_frame()
    payload = _dump(frame)
    # 把右半刚架推到远处并删掉横梁，形成两块
    payload["members"] = [m for m in payload["members"] if m["id"] != "b"]
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "DISCONNECTED_STRUCTURE"


def test_insufficient_support_http_error():
    frame, _ = make_portal_frame()
    payload = _dump(frame)
    # 松开两个柱底，整体无支承
    payload["nodes"][0]["restraints"] = [False, False, False]
    payload["nodes"][3]["restraints"] = [False, False, False]
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INSUFFICIENT_SUPPORT"


def test_bad_load_reference_http_error():
    frame, _ = make_portal_frame()
    payload = _dump(frame)
    payload["nodal_loads"][0]["node_id"] = "nope"
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INVALID_LOAD_REFERENCE"


def test_malformed_json_returns_structured_error():
    resp = client.post("/solve", content="{not valid json",
                       headers={"content-type": "application/json"})
    assert resp.status_code == 422
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == "INVALID_REQUEST"


def test_missing_required_field_returns_structured_error():
    resp = client.post("/solve", json={"nodes": []})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "INVALID_REQUEST"
    assert "message" in body["error"] and body["error"]["message"]


def test_openapi_documents_both_entrypoints():
    """对外暴露两个并列入口：/solve（静力）与 /stability（稳定）。"""
    spec = client.get("/openapi.json").json()
    assert "post" in spec["paths"]["/solve"]
    assert "post" in spec["paths"]["/stability"]


def test_solve_response_contract_unchanged():
    """静力入口的响应结构必须保持原样（本次升级不得改变 /solve 的输出）。"""
    frame, _ = make_portal_frame()
    resp = client.post("/solve", json=_dump(frame))
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"success", "displacements", "member_forces", "reactions"}
    assert body["success"] is True
    for d in body["displacements"]:
        assert set(d) == {"node_id", "ux", "uy", "theta"}
    for m in body["member_forces"]:
        assert set(m) == {"member_id", "node_i", "node_j", "end_i", "end_j", "axial_force"}
        for end in ("end_i", "end_j"):
            assert set(m[end]) == {"axial", "shear", "moment"}
    for r in body["reactions"]:
        assert set(r) == {"node_id", "fx", "fy", "moment", "restrained"}


# ---------------------------------------------------------------------------
# 稳定分析入口 /stability
# ---------------------------------------------------------------------------


def test_stability_success_shape():
    """欧拉柱（两端铰支，n=4）：响应结构 + 临界因子命中 π²EI/L²。"""
    frame, p = make_euler_column(n=4, bc="pinned")
    resp = client.post("/stability", json=_dump(frame))
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"success", "has_buckling_mode",
                         "critical_load_factor", "mode_shape"}
    assert body["success"] is True and body["has_buckling_mode"] is True
    euler = math.pi**2 * p["e"] * p["i"] / p["l"] ** 2
    assert body["critical_load_factor"] * p["p"] == pytest.approx(euler, rel=5e-3)
    assert len(body["mode_shape"]) == p["n"] + 1
    for m in body["mode_shape"]:
        assert set(m) == {"node_id", "ux", "uy", "theta"}
        assert all(math.isfinite(v) for k, v in m.items() if k != "node_id")
    # 归一化约定：绝对值最大的分量为 +1
    assert max(abs(m["ux"]) + abs(m["uy"]) + abs(m["theta"])
               for m in body["mode_shape"]) >= 1.0


def test_stability_tension_returns_no_buckling_mode_error():
    """受拉柱：422 + NO_BUCKLING_MODE，绝不硬凑一个数。"""
    frame, _ = make_euler_column(n=4, bc="pinned", tension=True)
    resp = client.post("/stability", json=_dump(frame))
    assert resp.status_code == 422
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == "NO_BUCKLING_MODE"
    assert body["error"]["message"]


def test_stability_invalid_model_reuses_static_error_codes():
    """稳定入口与静力入口共用同一套校验：非法模型返回同样的错误代码。"""
    frame, _ = make_euler_column(n=4)
    payload = _dump(frame)
    payload["members"] = payload["members"][:2]  # 删掉尾部杆件，结构断开
    resp = client.post("/stability", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "DISCONNECTED_STRUCTURE"

    payload = _dump(frame)
    payload["nodes"][0]["restraints"] = [False, False, False]  # 松开左端支座
    resp = client.post("/stability", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INSUFFICIENT_SUPPORT"


def test_stability_malformed_json_returns_structured_error():
    resp = client.post("/stability", content="{not valid json",
                       headers={"content-type": "application/json"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"

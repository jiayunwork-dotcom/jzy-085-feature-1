"""/buckling HTTP 入口测试（FastAPI TestClient）。

覆盖：
- 成功响应结构 success / has_valid_buckling_mode / critical_load_factor /
  buckling_mode，全部数值有限、无 NaN；
- 受拉主导 / 零荷载 -> HTTP 422 NO_POSITIVE_CRITICAL_FACTOR；
- 非法几何 / 荷载、坏 JSON -> 与 /solve 同款结构化错误；
- OpenAPI 同时登记 /solve 与 /buckling；
- /solve 的行为与响应结构不受新入口影响（回归保护）。
"""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from conftest import make_axial_column, make_portal_frame
from framesolver.main import app

client = TestClient(app)


def test_buckling_success_response_shape():
    frame, p = make_axial_column(n_segments=4, boundary="pinned-pinned")
    resp = client.post("/buckling", json=frame.model_dump())
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["has_valid_buckling_mode"] is True
    assert set(body) == {
        "success", "has_valid_buckling_mode",
        "critical_load_factor", "buckling_mode",
    }
    lam = body["critical_load_factor"]
    assert isinstance(lam, float) and math.isfinite(lam) and lam > 0.0
    # λ·P 命中欧拉临界力（0.2% 容差）
    assert lam * p["p"] == pytest.approx(p["euler_load"], rel=2.0e-3)
    assert len(body["buckling_mode"]) == len(frame.nodes)
    for m in body["buckling_mode"]:
        assert set(m) == {"node_id", "ux", "uy", "theta"}
        for key in ("ux", "uy", "theta"):
            assert isinstance(m[key], float) and math.isfinite(m[key])
    # 模态最大绝对分量归一化为 1
    peak = max(abs(m[k]) for m in body["buckling_mode"] for k in ("ux", "uy", "theta"))
    assert peak == 1.0


def test_buckling_tension_dominated_http_422():
    frame, _ = make_axial_column(n_segments=4, boundary="pinned-pinned")
    frame.nodal_loads[0].fy = +10.0
    resp = client.post("/buckling", json=frame.model_dump())
    assert resp.status_code == 422
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == "NO_POSITIVE_CRITICAL_FACTOR"
    assert body["error"]["message"]
    assert "NaN" not in resp.text


def test_buckling_zero_load_http_422():
    frame, _ = make_axial_column(n_segments=4, boundary="pinned-pinned")
    frame.nodal_loads = []
    resp = client.post("/buckling", json=frame.model_dump())
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "NO_POSITIVE_CRITICAL_FACTOR"


def test_buckling_invalid_property_http_error():
    frame, _ = make_axial_column(n_segments=2, boundary="pinned-pinned")
    payload = frame.model_dump()
    payload["members"][0]["inertia"] = -1.0
    resp = client.post("/buckling", json=payload)
    assert resp.status_code in (400, 422)
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] in ("INVALID_REQUEST", "INVALID_PROPERTY")


def test_buckling_ghost_node_http_error():
    frame, _ = make_axial_column(n_segments=2, boundary="pinned-pinned")
    payload = frame.model_dump()
    payload["members"][0]["node_j"] = "ghost"
    resp = client.post("/buckling", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "MEMBER_NODE_NOT_FOUND"


def test_buckling_bad_load_reference_http_error():
    frame, _ = make_axial_column(n_segments=2, boundary="pinned-pinned")
    payload = frame.model_dump()
    payload["nodal_loads"][0]["node_id"] = "nope"
    resp = client.post("/buckling", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INVALID_LOAD_REFERENCE"


def test_buckling_malformed_json_http_error():
    resp = client.post("/buckling", content="{not valid json",
                       headers={"content-type": "application/json"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_buckling_load_scaling_http_inverse_proportional():
    """HTTP 端到端：荷载翻倍 -> 临界因子减半。"""
    f1, _ = make_axial_column(n_segments=6, boundary="fixed-free", p=10.0)
    f2, _ = make_axial_column(n_segments=6, boundary="fixed-free", p=20.0)
    lam1 = client.post("/buckling", json=f1.model_dump()).json()["critical_load_factor"]
    lam2 = client.post("/buckling", json=f2.model_dump()).json()["critical_load_factor"]
    assert lam2 == lam1 / 2.0


def test_openapi_documents_both_entrypoints():
    spec = client.get("/openapi.json").json()
    assert "/solve" in spec["paths"] and "post" in spec["paths"]["/solve"]
    assert "/buckling" in spec["paths"] and "post" in spec["paths"]["/buckling"]


def test_solve_endpoint_unchanged_after_buckling_added():
    """新能力不能改动静力入口的输入输出：门式刚架响应结构逐字段钉死。"""
    frame, _ = make_portal_frame()
    resp = client.post("/solve", json=frame.model_dump())
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"success", "displacements", "member_forces", "reactions"}
    assert body["success"] is True
    assert len(body["displacements"]) == 4
    assert len(body["member_forces"]) == 3
    assert len(body["reactions"]) == 4
    for d in body["displacements"]:
        assert set(d) == {"node_id", "ux", "uy", "theta"}
    for m in body["member_forces"]:
        assert set(m) == {"member_id", "node_i", "node_j", "end_i", "end_j",
                          "axial_force"}
        assert set(m["end_i"]) == {"axial", "shear", "moment"}
        assert set(m["end_j"]) == {"axial", "shear", "moment"}
    for r in body["reactions"]:
        assert set(r) == {"node_id", "fx", "fy", "moment", "restrained"}
    # 静力解里绝不该冒出屈曲相关字段
    assert "critical_load_factor" not in body
    assert "buckling_mode" not in body
    assert "has_valid_buckling_mode" not in body

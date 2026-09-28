"""HTTP 入口：POST /solve（线性静力）与 POST /buckling（弹性屈曲）。

输入均为 FrameInput（JSON）；非法模型、奇异矩阵或不存在正临界因子时
统一返回 ErrorResponse：
``{"success": false, "error": {"code": ..., "message": ...}}``。
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .analysis import analyze_buckling, analyze_frame
from .errors import FrameError
from .models import (
    BucklingResponse,
    ErrorDetail,
    ErrorResponse,
    FrameInput,
    SolveResponse,
)

app = FastAPI(
    title="平面刚架直接刚度法核算服务",
    version="1.1.0",
    description="输入一副平面刚架：/solve 返回节点位移、杆端内力与支座反力；"
    "/buckling 返回弹性临界荷载因子与屈曲模态。",
)


def _error_payload(code: str, message: str) -> dict:
    return ErrorResponse(error=ErrorDetail(code=code, message=message)).model_dump()


@app.exception_handler(FrameError)
async def frame_error_handler(request: Request, exc: FrameError) -> JSONResponse:
    """业务错误：带机器可读代码与给人看的说明，不产生 500。"""
    return JSONResponse(status_code=exc.http_status, content=_error_payload(exc.code, exc.message))


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """请求体不符合 Schema（字段缺失 / 类型错误 / 非正数等）。"""
    details = "; ".join(
        f"{'.'.join(str(p) for p in err['loc'] if p != 'body') or '请求体'}: {err['msg']}"
        for err in exc.errors()
    )
    return JSONResponse(
        status_code=422,
        content=_error_payload("INVALID_REQUEST", f"请求 JSON 不符合模型定义：{details}"),
    )


@app.post(
    "/solve",
    response_model=SolveResponse,
    responses={
        400: {"model": ErrorResponse, "description": "非法模型"},
        422: {"model": ErrorResponse, "description": "请求格式错误或刚度矩阵奇异"},
    },
    summary="平面刚架受力核算",
)
def solve(frame: FrameInput) -> dict:
    """对一副平面刚架执行直接刚度法求解。"""
    return analyze_frame(frame)


@app.post(
    "/buckling",
    response_model=BucklingResponse,
    responses={
        400: {"model": ErrorResponse, "description": "非法模型"},
        422: {
            "model": ErrorResponse,
            "description": "请求格式错误、刚度矩阵奇异，或该荷载方向下不存在正临界因子",
        },
    },
    summary="平面刚架弹性屈曲（临界荷载因子）分析",
)
def buckling(frame: FrameInput) -> dict:
    """先做一阶线性分析取得轴力，再求解弹性稳定广义特征值问题。"""
    return analyze_buckling(frame)

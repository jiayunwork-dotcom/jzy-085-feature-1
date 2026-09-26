"""HTTP 入口：POST /solve（静力核算）与 POST /stability（弹性稳定分析）。

两个入口共用同一个 FrameInput 输入模型；
非法模型或数值失败统一返回 ErrorResponse：
``{"success": false, "error": {"code": ..., "message": ...}}``。
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .analysis import analyze_frame
from .errors import FrameError
from .models import ErrorDetail, ErrorResponse, FrameInput, SolveResponse, StabilityResponse
from .stability import analyze_stability

app = FastAPI(
    title="平面刚架直接刚度法核算服务",
    version="1.1.0",
    description="输入一副平面刚架，返回节点位移、杆端内力与支座反力（/solve），"
    "或弹性临界荷载因子与屈曲模态（/stability）。",
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
    "/stability",
    response_model=StabilityResponse,
    responses={
        400: {"model": ErrorResponse, "description": "非法模型"},
        422: {"model": ErrorResponse, "description": "请求格式错误、刚度矩阵奇异或不存在正临界因子"},
    },
    summary="平面刚架弹性稳定（屈曲）分析",
)
def stability(frame: FrameInput) -> dict:
    """对一副平面刚架执行两段式弹性稳定分析。

    先按当前荷载做一阶线性分析取得各杆轴力，再组装几何刚度并求解
    广义特征值问题，返回弹性临界荷载因子与屈曲模态。
    """
    return analyze_stability(frame)

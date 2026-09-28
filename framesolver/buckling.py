"""几何刚度（初应力刚度）与弹性临界荷载因子求解。

稳定分析分两段（见 ``analysis.analyze_buckling``）：第一段照常做一阶线性
静力分析，得到每根杆在**当前荷载**下的真实轴力；本模块负责第二段——
依据这些轴力逐杆形成几何刚度，求解广义特征值问题。

几何刚度的力学含义
==================

有轴力的梁，其切向刚度是“材料（弹性）刚度”与“几何刚度”之和：

    K_tan = K_e + K_g(N)

受压时（N<0，本服务约定拉力为正）轴力会制造随侧向位移增大而释放势能的
项，削弱抗弯 / 抗侧能力；受拉时（N>0）则相反，像绷紧的弦一样加固。
局部坐标下 6 阶几何刚度（自由度序与 ``element.py`` 完全一致：
[i 端轴向, i 端横向, i 端转角, j 端轴向, j 端横向, j 端转角]）：

- 轴向部分用线性形函数：N/L·[[1,-1],[-1,1]]（桁架几何刚度）；
- 弯曲部分用与弹性刚度同一套 Hermite 三次形函数积分
  ``∫ N·N'(x)ᵀN'(x) dx``（一致几何刚度，N 为轴力标量，拉正压负）：

        [ 36     3L   -36     3L]
        [ 3L   4L²   -3L    -L²]
    H = [-36    -3L    36    -3L]  · 1/(30L)
        [ 3L    -L²   -3L    4L²]

  这是初直梁 Total Lagrangian（Green 应变 ε = u′ + ½(v′)² 在基态
  v′=0 线性化）格式的经典一致几何刚度，轴向与弯曲之间无耦合块；
  单元刚体平移方向严格零能，单元刚体转动方向上存在初应力“自旋项”
  N·L·α²（frame-indifference 的已知项，与随动荷载刚度配对，且在
  结构上刚体转动已被支座约束消除），不影响屈曲特征值——离散加密后
  欧拉临界力 O(h²) 收敛于闭式解（见 tests/test_buckling.py）。

坐标变换与弹性刚度共用 ``geometry.py`` 的同一套约定：
``K_g_global = Tᵀ K_g_local T``，不另立方向约定。

特征值问题
==========

荷载整体放大 λ 倍时轴力也放大 λ 倍（一阶理论，轴力按当前内力线性外推），
切向刚度被吃穿（出现相邻平衡位形）的条件：

    det(K_e,f + λ K_g,f) = 0   ⟺   (K_e,f + λ K_g,f) φ = 0

即广义特征值问题 ``K_g φ = δ K_e φ``，临界因子 ``λ_cr = -1/δ``。
用 K_e 的 Cholesky 分解（K_e,f 对称正定）把它对称化：

    K_e = L Lᵀ,   H = L⁻¹ K_g L⁻ᵀ,   H u = δ u

H 对称，直接 ``eigh`` 即可，且与 ``K_e⁻¹ K_g`` 同特征值。
只有负的 δ 对应正的临界因子；δ ≥ 0 的根（含零根——刚体残留 / 纯受拉
加固方向）一律不是屈曲因子。
"""

from __future__ import annotations

import numpy as np

from .errors import NoBucklingError, SingularMatrixError

# 数值零根判据：对称矩阵特征值的绝对噪声界 eps·n·max|δ|（LAPACK 界），
# 再放宽两个量级留余量。它只滤掉舍入噪声（纯受拉结构里假负根为 eps 级），
# 不会误杀比轴向根小三个量级的真实弯曲屈曲根。
_EIGENVALUE_ZERO_FACTOR = 1.0e-12
# 特征向量能量归一化后的最大分量下限，低于它说明求解不可信
_MODE_RESIDUAL_TOL = 1.0e-8


def local_geometric_stiffness(length: float, axial_force: float) -> np.ndarray:
    """形成 6×6 局部坐标系单元几何刚度矩阵。

    参数
    ----
    length:       杆长 L
    axial_force:  第一段线性分析解出的本杆轴力 N，**拉力为正、压力为负**
                  （与 ``MemberForces.axial_force`` 同一约定）。
    """
    l = float(length)
    n = float(axial_force)
    kg = np.zeros((6, 6), dtype=float)

    # 轴向子块（自由度 0、3）：线性形函数
    axial_block = (n / l) * np.array([[1.0, -1.0], [-1.0, 1.0]])
    kg[np.ix_((0, 3), (0, 3))] = axial_block

    # 弯曲子块（自由度 1、2、4、5）：Hermite 一致几何刚度
    factor = n / (30.0 * l)
    l2 = l * l
    bending_block = factor * np.array(
        [
            [36.0, 3.0 * l, -36.0, 3.0 * l],
            [3.0 * l, 4.0 * l2, -3.0 * l, -l2],
            [-36.0, -3.0 * l, 36.0, -3.0 * l],
            [3.0 * l, -l2, -3.0 * l, 4.0 * l2],
        ]
    )
    idx = (1, 2, 4, 5)
    kg[np.ix_(idx, idx)] = bending_block
    return kg


def solve_buckling_eigenproblem(
    k_ff: np.ndarray, kg_ff: np.ndarray
) -> tuple[float, np.ndarray]:
    """在同一组自由自由度上求解弹性屈曲广义特征值问题。

    参数为已经过同一套“划行划列”约束缩聚的弹性刚度与几何刚度。
    返回 ``(最低正临界因子 λ_cr, 缩聚自由度上的屈曲模态向量)``；
    不存在正临界因子时抛 :class:`NoBucklingError`。
    """
    if not (np.all(np.isfinite(k_ff)) and np.all(np.isfinite(kg_ff))):
        raise SingularMatrixError("缩聚矩阵中出现非有限值（NaN 或无穷），无法做稳定分析")
    if k_ff.shape[0] == 0:
        raise NoBucklingError("结构没有任何自由自由度，不存在可发生屈曲的变形方向")

    k_sym = 0.5 * (k_ff + k_ff.T)
    g_sym = 0.5 * (kg_ff + kg_ff.T)

    # 第一段静力解已经在同一矩阵上做过 SVD 判秩；这里 Cholesky 是最后防线
    try:
        chol = np.linalg.cholesky(k_sym)
    except np.linalg.LinAlgError:
        raise SingularMatrixError(
            "弹性刚度矩阵不对称正定，无法形成广义特征值问题（结构存在机构）"
        )

    # H = L⁻¹ K_g L⁻ᵀ，与 K_e⁻¹ K_g 同特征值且保持对称
    eye = np.eye(chol.shape[0])
    l_inv = np.linalg.solve(chol, eye)
    h = l_inv @ g_sym @ l_inv.T
    h = 0.5 * (h + h.T)

    eig_values, eig_vectors = np.linalg.eigh(h)
    if not np.all(np.isfinite(eig_values)):
        raise SingularMatrixError("特征值中出现非有限值，广义特征值问题求解失败")

    # 数值零根判据：对称矩阵特征值的舍入噪声界 ≈ eps·n·max|δ|（LAPACK
    # 绝对误差界），再放宽两个量级留余量。纯受拉结构的“假负根”只是 eps
    # 级噪声，被滤掉；真实弯曲屈曲根（即便比轴向根小很多个量级）远大于该界。
    # 阈值只随谱尺度缩放（不加绝对下限），否则极小荷载下真根会被误杀。
    n_free = h.shape[0]
    max_abs_eig = float(np.max(np.abs(eig_values))) if n_free else 0.0
    if max_abs_eig == 0.0:
        raise NoBucklingError(
            "几何刚度在所有自由自由度方向上均为零（荷载为零或内力自平衡），"
            "在当前荷载方向下不存在弹性临界荷载因子"
        )
    zero_cut = _EIGENVALUE_ZERO_FACTOR * max(n_free, 1) * max_abs_eig
    negative = [(float(w), i) for i, w in enumerate(eig_values) if w < -zero_cut]

    if not negative:
        raise NoBucklingError(
            "在当前荷载方向下不存在正的弹性临界荷载因子：荷载为零，"
            "或结构在当前荷载下以受拉为主，几何刚度对所有变形方向都是加固的"
        )

    # δ 与 λ 的关系为 λ = -1/δ，故最负的 δ 对应最小的正临界因子
    delta, mode_index = min(negative, key=lambda item: item[0])
    lambda_cr = -1.0 / delta
    if not np.isfinite(lambda_cr) or lambda_cr <= 0.0:
        raise NoBucklingError("未能从特征值中恢复出正的临界荷载因子")

    # 模态从 u 换回原自由度：φ = L⁻ᵀ u（K_e φ = -λ K_g φ 的方向）
    u_vec = eig_vectors[:, mode_index]
    phi = np.linalg.solve(chol.T, u_vec)
    _check_mode_residual(k_sym, g_sym, phi, delta)

    return lambda_cr, phi


def _check_mode_residual(
    k_sym: np.ndarray, g_sym: np.ndarray, phi: np.ndarray, delta: float
) -> None:
    """残差复核：K_g φ 应等于 δ K_e φ。"""
    residual = g_sym @ phi - delta * (k_sym @ phi)
    scale = max(
        float(np.max(np.abs(g_sym @ phi))),
        abs(delta) * float(np.max(np.abs(k_sym @ phi))),
        1.0,
    )
    if float(np.max(np.abs(residual))) > _MODE_RESIDUAL_TOL * scale:
        raise SingularMatrixError("屈曲特征向量残差超出容差，结果不可信")

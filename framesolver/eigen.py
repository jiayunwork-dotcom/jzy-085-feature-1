"""广义特征值问题（弹性稳定 / 分岔屈曲）求解。

数学问题：当前荷载整体放大 λ 倍时，结构切线刚度奇异，

    (K_e + λ K_g) φ = 0   ⟺   K_e φ = λ (−K_g) φ

其中 K_e 为缩聚后的弹性刚度（对称正定），K_g 为按当前轴力（拉为正）
组装并**用同一组自由度、同一种划行划列方式缩聚**的几何刚度。
杆件受压时 K_g 半负定、−K_g 半正定，存在正特征值；
全部荷载乘 λ 等价于 K_g 乘 λ（轴力与荷载线性相关），
故最低的正特征值 λ_cr 即弹性临界荷载因子，对应特征向量为屈曲模态。

数值方法（只用 NumPy，全程保持对称形式）：

1. Cholesky 分解 K_e = L Lᵀ（K_e 已经过 solver.py 的正定性检验）；
2. 令 y = Lᵀ φ，化为标准对称特征值问题
       A y = (1/λ) y，  A = L⁻¹ (−K_g) L⁻ᵀ；
3. np.linalg.eigh 求全部特征对；最大的正特征值 ν_max 对应最低的
   正荷载因子 λ_cr = 1/ν_max，模态由 φ = L⁻ᵀ y 回代；
4. 非物理特征值甄别：ν ≤ 0（对应受拉主导或荷载反向才屈曲）与
   数值零（K_g 零空间，如纯轴向变形模态、无轴力杆件）一律丢弃；
5. 对选出的特征对强制残差复核，绝不把不可信的数字当结果返回。
"""

from __future__ import annotations

import numpy as np

from .errors import EigenSolutionError

# 判定“正特征值”的相对阈值（按谱范数归一）：
# 小于该阈值的正特征值对应天文数字般的临界因子，无工程意义，
# 且与 K_g 零空间给出的数值零（量级 1e-16）无法区分，一律视为零
_POSITIVE_REL_TOL = 1e-9
# 特征对残差的相对阈值
_RESIDUAL_REL_TOL = 1e-6


def lowest_positive_factor(
    k_e: np.ndarray, k_g: np.ndarray
) -> tuple[float | None, np.ndarray | None]:
    """求解 K_e φ = λ(−K_g) φ 的最低正特征值与对应特征向量。

    参数
    ----
    k_e: 缩聚后的弹性刚度矩阵（对称正定）
    k_g: 同一组自由度上缩聚后的几何刚度矩阵（轴力拉为正组装）

    返回
    ----
    (λ_cr, φ)：最低正临界荷载因子与对应特征向量（自由自由度空间）；
    不存在正特征值时返回 (None, None)。
    """
    if k_e.shape[0] == 0:
        return None, None
    if not (np.all(np.isfinite(k_e)) and np.all(np.isfinite(k_g))):
        raise EigenSolutionError("刚度矩阵中出现非有限值（NaN 或无穷），无法求解特征值问题")

    k_e_sym = 0.5 * (k_e + k_e.T)
    m = -0.5 * (k_g + k_g.T)  # M = −K_g，强制对称

    try:
        chol = np.linalg.cholesky(k_e_sym)
    except np.linalg.LinAlgError as exc:
        raise EigenSolutionError(
            "弹性刚度矩阵无法 Cholesky 分解（非正定），广义特征值问题无法求解"
        ) from exc

    # A = L⁻¹ M L⁻ᵀ（对称）
    a = np.linalg.solve(chol, np.linalg.solve(chol, m).T)
    a = 0.5 * (a + a.T)

    eigenvalues, eigenvectors = np.linalg.eigh(a)  # 升序
    scale = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    if scale == 0.0:
        # K_g ≡ 0：结构在当前荷载下没有任何轴向力，谈不上屈曲
        return None, None

    # 最大特征值即最大的正 ν = 1/λ；非正（≤ 阈值）即无物理意义的正荷载因子
    nu_max = float(eigenvalues[-1])
    if nu_max <= _POSITIVE_REL_TOL * scale:
        return None, None

    factor = 1.0 / nu_max
    mode = np.linalg.solve(chol.T, eigenvectors[:, -1])

    # ---- 残差复核：K_e φ 应等于 λ (−K_g) φ ------------------------------
    if not np.all(np.isfinite(mode)):
        raise EigenSolutionError("特征向量中出现非有限值（NaN 或无穷），结果不可信")
    residual = k_e_sym @ mode - factor * (m @ mode)
    norm = max(float(np.max(np.abs(k_e_sym @ mode))), 1.0)
    if float(np.max(np.abs(residual))) > _RESIDUAL_REL_TOL * norm:
        raise EigenSolutionError("特征解残差超出容差，广义特征值求解结果不可信")

    return factor, mode

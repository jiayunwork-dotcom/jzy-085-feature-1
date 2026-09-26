"""弹性稳定（屈曲）分析：几何刚度与广义特征值求解。

数学模型
--------
线性静力分析回答「给定荷载下结构变形成什么样」；稳定分析回答
「把当前这套荷载整体放大多少倍，结构会冒出另一个能量上不分高下的
相邻平衡位形」（分岔失稳）。杆件轴力会改变其抗侧、抗弯能力：
受压削弱、受拉加固。把这份随内力变化的**几何刚度** K_g 按当前
轴力装出来，临界状态满足

    det(K_e + λ K_g) = 0     <=>     (K_e + λ K_g) φ = 0

即广义特征值问题 K_e φ = λ (−K_g) φ。最低的正特征值 λ_cr 就是
临界荷载因子（参考荷载整体放大 λ_cr 倍时失稳），对应特征向量
φ 为屈曲模态。

单元几何刚度（一致几何刚度，由 Hermite 形函数积分 N·(v′)² 得到）
----------------------------------------------------------------
局部坐标 6 阶矩阵，自由度顺序与弹性单元刚度完全一致
[i 端轴向, i 端横向, i 端转角, j 端轴向, j 端横向, j 端转角]，
只含横向 / 转角分量，轴向分量为零。N 为杆件轴力（**拉力为正**，
与 forces.py 的 axial_force 约定一致）：

         N       [ 0   0      0       0   0      0      ]
    K_g = ─      [ 0   6/5    L/10    0  -6/5    L/10   ]
         30L     [ 0   3L     4L²     0  -3L    -L²     ]
                 [ 0   0      0       0   0      0      ]
                 [ 0  -36    -3L      0   36    -3L     ]
                 [ 0   3L    -L²      0  -3L     4L²    ]

（写成 N/L 系数即横向对角 6N/5L、转角对角 2NL/15、耦合 −NL/30。）
受拉（N>0）时横向 / 转角方向上 K_g 为正定贡献（加固）；
受压（N<0）时为负定贡献（削弱），削弱到与 K_e 抵消即失稳。

特征值求解
----------
K_e（缩聚后）对称正定，做 Cholesky 分解 K_e = L Lᵀ，把广义问题
化为对称标准问题：令 φ = L⁻ᵀ ψ，则

    M ψ = (1/λ) ψ,   M = −L⁻¹ K_g L⁻ᵀ（对称）

M 的特征值 ω = 1/λ。正 ω 对应正临界因子（沿当前荷载方向继续
加载会失稳）；负 ω 对应「荷载反向后才失稳」；零 ω 对应不产生
几何刚度能量的模态（如纯轴向变形、零荷载）。只取正 ω，
λ_cr = 1/ω_max。若不存在正 ω，说明该荷载方向下结构以受拉为主，
不发生弹性屈曲，由调用方转成结构化错误。
"""

from __future__ import annotations

import numpy as np

from .errors import NoBucklingModeError, SingularMatrixError

# 判定「正特征值」的相对阈值：ω > _EIGEN_TOL × max|ω| 才算物理上的正。
# 用于剔除数值零（纯轴向模态、零荷载）与负值（受拉主导模态）。
_EIGEN_TOL = 1e-9


def local_geometric_stiffness(length: float, axial_force: float) -> np.ndarray:
    """形成 6×6 局部坐标系单元几何刚度矩阵。

    参数
    ----
    length:      杆长 L
    axial_force: 杆件轴力 N，**拉力为正**（受压取负值，削弱抗侧刚度）
    """
    l = float(length)
    n_l = float(axial_force) / (30.0 * l)  # N / (30L)

    kg = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 36.0, 3.0 * l, 0.0, -36.0, 3.0 * l],
            [0.0, 3.0 * l, 4.0 * l * l, 0.0, -3.0 * l, -l * l],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -36.0, -3.0 * l, 0.0, 36.0, -3.0 * l],
            [0.0, 3.0 * l, -l * l, 0.0, -3.0 * l, 4.0 * l * l],
        ],
        dtype=float,
    )
    return n_l * kg


def solve_critical_load(
    k_ff: np.ndarray, kg_ff: np.ndarray
) -> tuple[float, np.ndarray]:
    """求解 (K_e + λ K_g) φ = 0，返回 (最低正临界因子 λ_cr, 对应屈曲模态)。

    参数为**同一组自由自由度、同一种划行划列方式**下缩聚好的弹性刚度
    与几何刚度矩阵。模态按自由自由度给出，未做归一化（由调用方决定）。

    无正特征值（受拉主导 / 零荷载）时抛 :class:`NoBucklingModeError`；
    K_e 不正定或出现非有限值时抛 :class:`SingularMatrixError`。
    """
    if not (np.all(np.isfinite(k_ff)) and np.all(np.isfinite(kg_ff))):
        raise SingularMatrixError("刚度矩阵中出现非有限值（NaN 或无穷），无法做稳定分析")

    n = k_ff.shape[0]
    if n == 0:
        raise NoBucklingModeError("结构没有自由自由度，不存在可失稳的变形模态")

    # 强制对称，消除组装中量级 1e-16 的数值不对称
    k_sym = 0.5 * (k_ff + k_ff.T)
    g_sym = 0.5 * (kg_ff + kg_ff.T)

    # K_e 与静力段求解的是同一个矩阵，理论对称正定；
    # Cholesky 失败说明矩阵病态 / 非正定，按奇异处理
    try:
        chol = np.linalg.cholesky(k_sym)
    except np.linalg.LinAlgError as exc:
        raise SingularMatrixError(
            "弹性刚度矩阵非正定，无法形成稳定分析特征值问题（结构存在机构或零能模态）"
        ) from exc

    # 广义问题 K_e φ = λ(−K_g)φ 化为对称标准问题 M ψ = (1/λ) ψ，
    # 其中 M = −L⁻¹ K_g L⁻ᵀ，φ = L⁻ᵀ ψ
    y = np.linalg.solve(chol, g_sym)          # Y = L⁻¹ K_g
    m = -np.linalg.solve(chol, y.T).T         # M = −Y L⁻ᵀ = −L⁻¹ K_g L⁻ᵀ
    m = 0.5 * (m + m.T)                       # 数值上再保险一次对称性

    omegas, psis = np.linalg.eigh(m)
    if not np.all(np.isfinite(omegas)):
        raise SingularMatrixError("稳定分析特征值求解产生非有限值，结果不可信")

    omega_max_abs = float(np.max(np.abs(omegas)))
    if omega_max_abs == 0.0:
        # K_g ≡ 0：荷载不产生任何轴力（如零荷载），特征值问题退化
        raise NoBucklingModeError(
            "该荷载方向下不发生弹性屈曲：荷载未在杆件中产生轴力，"
            "几何刚度为零，无法定义临界荷载因子"
        )
    positive = omegas > _EIGEN_TOL * omega_max_abs
    if not np.any(positive):
        raise NoBucklingModeError(
            "该荷载方向下不发生弹性屈曲：结构内力以受拉为主（或荷载不产生有效轴压），"
            "广义特征值问题不存在正特征值，即没有正的临界荷载因子"
        )

    # 最低正 λ = 最大正 ω 的倒数
    idx = int(np.argmax(np.where(positive, omegas, -np.inf)))
    lambda_cr = 1.0 / float(omegas[idx])
    phi_free = np.linalg.solve(chol.T, psis[:, idx])

    if not np.isfinite(lambda_cr) or not np.all(np.isfinite(phi_free)):
        raise SingularMatrixError("临界荷载因子或屈曲模态出现非有限值，结果不可信")

    return lambda_cr, phi_free


def normalize_mode_shape(phi: np.ndarray) -> np.ndarray:
    """屈曲模态归一化：最大绝对分量 = 1，且该分量取正号（符号确定）。

    屈曲模态只定义形状、不定义幅值与正负号；固定这个约定让输出
    可复现、可比较。零向量（理论不应出现）原样返回。
    """
    phi = np.asarray(phi, dtype=float)
    scale = float(np.max(np.abs(phi))) if phi.size else 0.0
    if scale == 0.0:
        return phi
    normalized = phi / scale
    pivot = int(np.argmax(np.abs(normalized)))
    if normalized[pivot] < 0.0:
        normalized = -normalized
    return normalized

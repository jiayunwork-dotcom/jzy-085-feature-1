# 平面刚架受力核算服务（framesolver）

喂一副平面刚架的几何 / 截面 / 荷载 JSON，提供两个并列的分析类型：

- `POST /solve`：用**直接刚度法**解出节点位移，再回代出每根杆的轴力、
  剪力、弯矩和各支座反力（一阶线性静力）；
- `POST /buckling`：先做一阶线性分析取得各杆真实轴力，再求解弹性刚度与
  几何刚度构成的**广义特征值问题**，给出整体弹性临界荷载因子（分岔失稳）
  与对应的屈曲模态形状。

不做画图、不做进度台账，不设账户体系。

- 语言：Python 3.12（锁定）
- Web 框架：FastAPI
- 线性代数：NumPy

---

## 一条命令跑起来

### 方式一：Docker Compose（推荐）

```bash
docker compose up --build -d
# 服务在 http://localhost:8000
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d @examples/portal_frame.json
# 弹性屈曲（临界荷载因子 + 屈曲模态）
curl -X POST http://localhost:8000/buckling \
  -H "Content-Type: application/json" \
  -d @examples/euler_column.json
```

### 方式二：Docker

```bash
docker build -t frame-solver .
docker run --rm -p 8000:8000 frame-solver
```

### 方式三：本地 Python（3.12）

```bash
pip install -r requirements.txt
uvicorn framesolver.main:app --host 0.0.0.0 --port 8000
```

交互式接口文档（FastAPI 自带，仅用于调试）：`http://localhost:8000/docs`。

### 跑测试

```bash
pip install -r requirements-dev.txt
pytest
```

---

## 接口

### `POST /solve`

请求体：

```json
{
  "nodes": [
    {"id": "1", "x": 0.0, "y": 0.0, "restraints": [true, true, true]}
  ],
  "members": [
    {"id": "m1", "node_i": "1", "node_j": "2",
     "elastic_modulus": 2.1e8, "area": 2.0e-2, "inertia": 8.0e-5}
  ],
  "nodal_loads": [
    {"node_id": "2", "fx": 10.0, "fy": 0.0, "moment": 0.0}
  ],
  "distributed_loads": [
    {"member_id": "m1", "load_type": "uniform_transverse", "qy": -3.0}
  ]
}
```

- 每个节点 3 个自由度：水平 `u`、竖向 `v`、转角 `θ`；
  `restraints` 按此顺序给布尔值，`true` = 锁死。
  固定端 `[t,t,t]`、铰支座 `[t,t,f]`、水平向滚动支座（仅竖向链杆）`[f,t,f]`。
- 杆件带弹性模量、截面积、惯性矩（都必须为正）。
- 荷载两类：
  - 节点集中力 / 力矩（整体坐标，`moment` 逆时针为正）；
  - 杆件满跨、**垂直于杆轴**的均布荷载（局部坐标 `qy`，沿局部 +ȳ 为正，重力向下给负值）。

成功响应 `200`：

```json
{
  "success": true,
  "displacements": [{"node_id": "1", "ux": 0.0, "uy": 0.0, "theta": 0.0}],
  "member_forces": [{
    "member_id": "m1", "node_i": "1", "node_j": "2",
    "end_i": {"axial": 0.0, "shear": 0.0, "moment": 0.0},
    "end_j": {"axial": 0.0, "shear": 0.0, "moment": 0.0},
    "axial_force": 0.0
  }],
  "reactions": [{"node_id": "1", "fx": 0.0, "fy": 0.0, "moment": 0.0,
                 "restrained": [true, true, true]}]
}
```

失败响应（HTTP 400 / 422）：

```json
{"success": false, "error": {"code": "SINGULAR_MATRIX", "message": "……可读说明……"}}
```

| code | HTTP | 含义 |
|---|---|---|
| `DUPLICATE_NODE_ID` | 400 | 节点编号重复 |
| `MEMBER_NODE_NOT_FOUND` | 400 | 杆件端点指向不存在的节点 |
| `ZERO_LENGTH_MEMBER` | 400 | 杆长为零（两端坐标重合） |
| `INVALID_PROPERTY` | 400 | E / A / I 非正数，或坐标、荷载不是有限数 |
| `INVALID_LOAD_REFERENCE` | 400 | 荷载挂在不存在的节点 / 杆件上 |
| `DISCONNECTED_STRUCTURE` | 400 | 结构被分成互不相连的几块，或存在孤立节点 |
| `INSUFFICIENT_SUPPORT` | 400 | 约束不足以消除 3 个整体刚体运动 |
| `SINGULAR_MATRIX` | 422 | 缩聚刚度矩阵奇异（机构 / 零能模态） |
| `NO_POSITIVE_CRITICAL_FACTOR` | 422 | 荷载为零或受拉主导，该荷载方向下不发生弹性屈曲 |
| `INVALID_REQUEST` | 422 | 请求 JSON 不符合 Schema（字段缺失、类型错等） |

任何情况下都不会返回 NaN，也不会悄悄丢弃杆件：出错即结构化报错。

---

### `POST /buckling`

输入与 `/solve` **完全相同**（同一套几何 / 截面 / 约束 / 荷载模型）。
服务内部分两段：

1. 先按 `/solve` 同一条流水线做一阶线性静力分析，取得每根杆在当前荷载下
   的真实轴力（拉正压负）；
2. 用这些轴力逐杆形成**几何刚度**（受压削弱、受拉加固抗侧 / 抗弯能力），
   组装后与弹性刚度在**同一组自由度、同一种划行划列约束**下缩聚，求解
   广义特征值问题 `(K_e,f + λ K_g,f) φ = 0`。

成功响应 `200`：

```json
{
  "success": true,
  "has_valid_buckling_mode": true,
  "critical_load_factor": 1036.84,
  "buckling_mode": [
    {"node_id": "1", "ux": 0.0, "uy": 0.0, "theta": 0.0}
  ]
}
```

- `critical_load_factor`：当前这套荷载整体放大多少倍后发生整体屈曲。
  它与施加荷载成反比：荷载翻倍、因子减半（临界的是荷载总量）；
- `buckling_mode`：失稳瞬间的相邻平衡位形形状（屈曲模态），按全部自由度
  最大绝对值归一化为 1，受约束自由度为零；
- 特征谱中的非物理零根 / 正根（刚体残留、纯受拉加固方向）被甄别丢弃，
  只报告有物理意义的最低正临界因子。

不存在正临界因子（荷载为零，或结构在该荷载方向上以受拉为主、任何正
放大倍数都不会屈曲）时返回 `422`：

```json
{"success": false,
 "error": {"code": "NO_POSITIVE_CRITICAL_FACTOR",
           "message": "在当前荷载方向下不存在正的弹性临界荷载因子：……"}}
```

非法几何 / 荷载、结构为机构等情形复用与 `/solve` 完全相同的错误代码
（`INVALID_PROPERTY`、`SINGULAR_MATRIX` 等），绝不返回 NaN。

---

## 求解方法（直接刚度法）

1. **单元刚度**（`element.py`）：每根杆在局部坐标下形成同时含轴向与弯曲的
   6 阶刚度矩阵 `K_l`（Hermite 梁单元）。
2. **坐标变换**（`geometry.py`）：局部 x̄ 由 i 端指向 j 端，
   整体刚度 `K_g = Tᵀ K_l T`，`T = diag(R,R)`，
   `R = [[c,s,0],[-s,c,0],[0,0,1]]`。
3. **总刚组装**（`assembly.py`）：按自由度 scatter-add 进总刚，
   节点荷载与均布荷载等效节点力（`[0, qL/2, qL²/12, 0, qL/2, −qL²/12]`）
   叠加进荷载向量。
4. **约束施加**（`constraints.py`）：全程只使用**划行划列法**一种——
   被约束自由度位移置零并从方程中删去，解 `K_ff d_f = P_f`；
   不使用大数罚函数，因而没有罚系数导致的病态问题。
5. **方程求解**（`solver.py`）：SVD 判秩（最小奇异值 < 1e-10 × 最大奇异值
   即判奇异），Cholesky 求解（失败退对称 SVD），并强制残差复核（1e-8）。
6. **内力回代**（`forces.py`）：杆端力严格用**本杆自己的局部刚度乘本杆的
   局部位移**恢复：`f = P_eq − K_l (T d_g)`；不另起弯矩查表。
7. **反力回代**：取受约束行 `R = (K d − P)_r`。

### 稳定分析（/buckling）两段法

1. **第一段（线性静力）**：与上面完全相同的直接刚度法，取每根杆 i 端
   局部轴向分量作为轴力 `N`（拉正压负）。几何刚度必须来自这份真实解出的
   轴力，不做任何假定。
2. **单元几何刚度**（`buckling.py`）：局部 6 阶，自由度顺序与 `element.py`
   一致。轴向块 `N/L·[[1,−1],[−1,1]]`（线性形函数）；弯曲块用同一套
   Hermite 三次形函数积分 `∫ N N'ᵀN' dx`（一致几何刚度）：

       [ 36    3L  −36    3L]
       [ 3L  4L²  −3L   −L²]
       [−36   −3L   36   −3L] · N/(30L)
       [ 3L   −L²  −3L  4L²]

   轴力为负（受压）时整块削弱抗侧 / 抗弯，为正（受拉）时加固；坐标变换
   与弹性刚度共用 `geometry.py` 同一套约定：`K_g,global = Tᵀ K_g,local T`。
3. **同组自由度缩聚**：整体几何刚度与弹性刚度在**完全相同**的自由自由度
   上以同一种划行划列法缩聚（不与罚函数混用），解
   `det(K_e,f + λ K_g,f) = 0`。
4. **广义特征值**：`K_e,f = L Lᵀ` Cholesky 对称化为
   `H = L⁻¹ K_g,f L⁻ᵀ`，对对称阵 `H` 直接 `eigh`。δ<0 的根给出正因子
   `λ = −1/δ`，取**最负**的 δ（= 最小正 λ）作为临界因子；eps·n 量级的
   数值零根与正根一律丢弃。特征向量换回原自由度 `φ = L⁻ᵀ u`，做切向
   刚度残差复核后全自由度归一化输出。

> 几何刚度取初直梁 Total Lagrangian 一致形式（Green 应变在基态线性化，
> 轴弯无耦合块）：单元刚体平移严格零能；单元刚体转动方向上的初应力
> “自旋项”在结构上随刚体转动被支座消除，不进入屈曲根，欧拉临界力随
> 分段加密 O(h²) 单调收敛于闭式解（见下节算例）。

### 符号约定（测试与输出一致）

- 整体坐标 x 向右、y 向上，转角 / 弯矩逆时针为正；
- 杆端力为**杆件作用于节点**的局部坐标六维力；
- `axial_force` 取 i 端轴向分量，**拉力为正**；j 端轴向分量与之等值反号；
- 标量截面弯矩（测试里用于和教材对照）以**下侧受拉为正**。

---

## 手算校核算例（门式刚架）

见 `examples/portal_frame.json`，也是 `tests/test_portal_frame.py` 的算例：

- 两柱底**固定**，柱高 = 梁跨 = L = 4 m，柱梁等截面（E = 2.1×10⁸ kN/m²、
  I = 8×10⁻³ m⁴、A 取 2×10⁴ m² 使轴向变形可忽略）；
- 横梁高度处（左柱顶节点 2）作用水平集中力 P = 10 kN（+x 向）。

按位移法（忽略轴向变形，2 个未知量：层间侧移 Δ、节点转角 θ）的闭式解：

| 量 | 公式 | 数值 | 服务输出（有限 A） |
|---|---|---|---|
| 柱顶侧移 | 5PL³/(84EI) | 2.2676×10⁻⁵ m | 2.2676×10⁻⁵ m |
| 节点转角 | −PL²/(28EI) | −3.4014×10⁻⁶ rad | −3.4014×10⁻⁶ rad |
| 水平反力 | 各 −P/2 | −5.0 kN | −5.0 kN |
| 竖向反力 | ∓3P/7 | ∓4.2857 kN | −4.2857 / +4.2857 |
| 柱底弯矩 | 2PL/7 | 11.4286 kN·m | 11.4286 kN·m |
| 梁端弯矩 | 3PL/14 | 8.5714 kN·m | 8.5714 kN·m |
| 横梁跨中弯矩 | 反弯点，0 | 0 | 0 |

另附梁的经典解测试：两端固定梁均布荷载 q 的固端弯矩 qL²/12、跨中 qL²/24；
简支梁跨中 qL²/8、反力各 qL/2。

---

## 手算校核屈曲算例（欧拉柱）

见 `examples/euler_column.json`（竖直等截面柱，E = 2.1×10⁸、
A = 2×10⁻²、I = 8×10⁻⁵、L = 4 m，顶端轴心压力 P = 10 kN），
也是 `tests/test_buckling.py` 的算例。欧拉闭式临界力

```
P_cr = π² E I / L₀²
```

| 两端约束 | 计算长度 L₀ | 欧拉 P_cr (kN) | 服务 λ·P（4 段） | 相对误差 |
|---|---|---|---|---|
| 两端铰接 | L | 10363.08 | 10368.39 | +0.051% |
| 一端固定一端自由 | 2L | 2590.77 | 2590.86 | +0.003% |

离散加密时相对误差单调收敛（两端铰接：2 段 0.75% → 4 段 0.051% →
8 段 0.0033% → 16 段 0.0002%），符合一致几何刚度的 O(h²) 上侧收敛。
同一模型把荷载翻倍，临界因子恰好减半（`λ·P` 不变）；荷载反号（受拉）
时返回 `NO_POSITIVE_CRITICAL_FACTOR` 而不是硬凑一个数。

---

## 被自动化测试守死的力学恒等式

容差写成具体数值（力 / 弯矩 1×10⁻⁸；与忽略轴向变形的教材解比较用 0.2%
相对容差），见 `tests/`：

1. 全部支座反力 + 全部外荷载（含杆件均布荷载合力），整体坐标
   ΣFx = 0、ΣFy = 0；
2. 对**任一固定点**的合力矩为 0（测试取原点和两个任意点）；
3. 同一根杆两端轴力等值反号（含杆件隔离体六维自平衡）；
4. 每个刚性节点处，交汇各杆杆端弯矩 + 外加节点力矩 + 支座反力矩 = 0
   （水平、竖向力平衡同样逐节点检查）。

测试共 102 个，分布在：

- `test_elements.py`：方向余弦、变换正交性、单元刚度系数、刚体零空间、
  等效荷载合力、组装 scatter-add；
- `test_equilibrium.py`：上述四条力学恒等式；
- `test_portal_frame.py`：门式刚架教材反力 / 位移 / 弯矩与梁经典解；
- `test_validation.py`：全部非法输入类型与奇异性；
- `test_http_api.py`：/solve 的 HTTP 成功 / 各类错误 / 坏 JSON / OpenAPI；
- `test_buckling.py`：欧拉柱闭式解（铰接 L、悬臂 2L）与分段加密 O(h²)
  收敛、荷载反比例缩放、纯拉 / 纯压几何刚度加减号、自旋项取值、
  屈曲模态形状与切向刚度残差、受拉主导 / 零荷载无正因子、非法输入；
- `test_buckling_api.py`：/buckling 的 HTTP 响应结构、错误码、
  OpenAPI 登记，以及 /solve 行为与响应结构不被改动的回归保护。

---

## 模块划分

```
framesolver/
├── geometry.py     # 方向余弦与坐标变换
├── element.py      # 六阶局部单元刚度 + 均布荷载等效节点力
├── buckling.py     # 六阶局部几何刚度（一致形式）+ 屈曲广义特征值求解
├── assembly.py     # 总刚与荷载向量组装
├── constraints.py  # 自由度分类（划行划列法）
├── solver.py       # SVD 判秩 + Cholesky 求解 + 残差复核
├── forces.py       # 杆端内力与支座反力回代
├── validation.py   # 全部输入校验（含连通性、刚体约束）
├── models.py       # Pydantic 输入 / 输出模型（静力与屈曲各一套响应）
├── analysis.py     # 一阶线性分析（两入口共用）+ 静力 / 屈曲两条核算串联
└── main.py         # HTTP 入口 /solve 与 /buckling（FastAPI）
```

各文件互相独立、各管一段，便于阅读与维护。

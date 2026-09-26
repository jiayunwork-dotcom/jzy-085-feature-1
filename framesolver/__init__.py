"""平面刚架直接刚度法核算服务（planar rigid-frame solver）。

模块划分（各管一段、互不重叠）：

- geometry.py  方向余弦与坐标变换
- element.py   局部坐标系下带轴向 + 弯曲的六阶单元刚度、一致几何刚度、
               分布荷载等效节点力
- assembly.py  总刚 / 几何总刚与荷载向量组装
- constraints.py  自由度分类（受约束 / 自由），本实现统一采用“划行划列”法
- solver.py    方程求解与奇异性判定
- eigen.py     广义特征值问题（弹性稳定）求解
- forces.py    杆端内力与支座反力回代
- validation.py  输入校验（编号、拓扑、几何、材料、连通性、刚体约束）
- models.py    HTTP 输入 / 输出的 Pydantic 数据模型
- analysis.py  第一阶段线性静力分析 + 静力核算输出
- stability.py 两段式弹性稳定（屈曲）分析
- main.py      HTTP 入口（FastAPI）：/solve 与 /stability
"""

from .analysis import analyze_frame
from .stability import analyze_stability

__all__ = ["analyze_frame", "analyze_stability"]

# 多目标光谱仪光纤臂分配裁决服务

为天文台多目标光谱仪联合分配可伸缩光纤臂的 HTTP 服务。在**机械臂互不碰撞**
（任意两条"基座→目标"闭线段净距不小于给定安全距离）的硬约束下，按以下
**字典序**目标求全局最优分配，而不是逐臂贪心选择最近天体：

1. **最大化分配数量**；
2. 在此前提下**最大化已分配目标优先级之和**；
3. 再**最小化各臂伸长平方和**；
4. 最后选择**稳定分配序列**（按提交的臂顺序，优先让靠前的臂被分配，
   并分配到序号更小的目标）。

几何距离全程使用 `fractions.Fraction` **精确有理运算**，净距判定（含等号
边界）不依赖浮点容差。求解器将整个字典序目标编码为单个整数边权，用
Hungarian 最大权二分匹配求"忽略碰撞"的松弛上界，再对冲突边做
分支定界（包含/排除二分 + 记忆化），保证全局最优且确定性可复现。

## 目录

```
app/geometry.py   精确闭线段距离（含垂足/端点/退化情形）与证据点
app/solver.py     字典序目标编码、Hungarian、碰撞分支定界、证据组装
app/schemas.py    严格请求/响应模型（非法输入 422，不进入求解）
app/main.py       FastAPI：/health/live、/health/ready、裁决接口
tests/            几何对拍、求解器目标层级、端到端 API 测试
scripts/         健康探针、容器冒烟、一次性 verify 流程
Dockerfile        镜像（含容器级健康检查）
docker-compose.yml api 服务 + 一次性 verify 服务
```

## 快速开始（Docker Compose）

```bash
# 宿主机端口可配置（默认 8080）
HOST_PORT=9090 docker compose up --build -d api
curl -s http://localhost:9090/health/ready
```

健康检查只有在应用启动自检（几何核 + 一次最小规模端到端求解）通过、裁决
接口可以接收请求后才返回 healthy。

一次性验证（等待 api 健康后执行**代码测试 + 应用构建检查 + 含碰撞约束的
API 冒烟**，自行退出并以退出码汇总成败）：

```bash
docker compose up --build --abort-on-container-exit verify
# 或： docker compose run --rm verify
echo $?   # 0 表示全部通过
```

## 接口

`POST /api/v1/adjudicate`

请求（整数基座坐标、整数最大伸长、正整数优先级/净距/最低分配数；6–12 根
臂、6–16 个唯一目标）：

```json
{
  "arms": [
    {"id": "A0", "x": 0,   "y": 0, "max_extension": 500},
    {"id": "A1", "x": 100, "y": 0, "max_extension": 500}
  ],
  "targets": [
    {"id": "T0", "x": 10,  "y": 10, "priority": 1},
    {"id": "T1", "x": 110, "y": 10, "priority": 2}
  ],
  "clearance": 5,
  "minimum_allocations": 2
}
```

约束与行为：

* 每根臂至多连接一个**可达**目标（基座到目标距离 ≤ `max_extension`）；
* 目标不得复用；
* 任意两条已分配"基座→目标"**闭线段**距离必须 ≥ `clearance`；
* 输入数量越界、非正整数、重复 id、浮点/布尔坐标、多余字段等一律 `422`，
  非法输入**不会进入求解器**。

成功响应（节选）包含：

* `assignments`：`arm_id / target_id / extension / extension_sq` 配对；
* `unassigned_arms`、`unassigned_targets`：未分配对象；
* `arm_lengths`：各臂实际伸长；
* `clearance.pair_evidence`：**逐对最小净距证据**（精确值 `distance_exact`、
  平方精确值、浮点值、要求净距、是否满足、最近点见证坐标）；
* `objective`：分配数、优先级和、伸长平方和与优化顺序说明。

当达不到 `minimum_allocations` 时，接口仍返回 `200`，但
`status = "minimum_not_met"`、`feasible = false`，并给出：

* `objective.num_allocations`：**实际可达的最大分配数量**；
* `reason`：明确的自然语言原因（净距冲突 / 不可达 / 容量不足）；
* `witness.blocking_pair`：一对各自可达但互相违反净距的分配及其精确距离、
  最近点见证（若导致短缺的结构性冲突存在）；
* 返回的部分分配本身仍然满足全部碰撞约束并附完整逐对证据。

## 本地开发

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest -q
uvicorn app.main:app --port 8080
```

## 正确性验证

* `tests/test_geometry.py`：2000 组随机整数线段与浮点参考实现对拍，
  并校验见证点恰在两线段上、净距边界精确成立；
* `tests/test_solver.py`：目标四层级（数量→优先级→伸长²→稳定序列）、
  交叉/汇聚碰撞、不可达、容量上限等；
* `scripts/fuzz_compare.py`：数百个小实例与**全枚举暴力最优解**逐一对比；
* `tests/test_api.py`：严格输入校验、可行/短缺/证据、最大规模实例。

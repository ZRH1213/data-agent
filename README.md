# MovieLens 1M 数据治理与分析 Agent（迭代一、二）

本项目根据课程迭代要求实现。用户在 Streamlit 页面提交自然语言请求，DeepSeek 通过 API 选择工具；迭代一工具调用 Hadoop Streaming 完成原始评分、清洗和再次评分，迭代二工具训练/查询分类、聚类和降维模型。程序不会用预设评分或模拟异常代替执行结果。

**公开仓库只包含源码、测试和文档。** API 密钥、MovieLens 原始压缩包、清洗后的记录、模型、日志和 Agent 提示词/调用轨迹都不提交。`汇报辅助文档*.md` 中的数值是作者在本地修改过的输入上得到的历史演示快照，不是新部署的默认输出；使用官方数据重新运行时，行数和指标会改变。

## 从新电脑部署

1. 安装 Python 3.11+ 和 Docker Engine（Windows/macOS 可用 Docker Desktop，选择 Linux 容器）。本项目完整迭代一通过本机 Docker 守护进程运行 Hadoop；**没有 Docker 的托管 Streamlit 环境无法直接运行完整管线**。首次运行需能拉取 `apache/hadoop:3.4.1` 与 `python:3.11-slim`，程序会构建 `movielens-hadoop-python:3.4.1`。
2. 克隆仓库并安装依赖：

   ```powershell
   git clone https://github.com/ZRH1213/data-agent.git
   cd data-agent
   python -m pip install -r requirements.txt
   ```

3. 从 [GroupLens 官方 MovieLens 1M 页面](https://grouplens.org/datasets/movielens/1m/)自行取得 `ml-1m.zip`，放到仓库根目录。也可在仓库目录下载官方压缩包：

   ```powershell
   Invoke-WebRequest 'https://files.grouplens.org/datasets/movielens/ml-1m.zip' -OutFile 'ml-1m.zip'
   ```

   Linux/macOS 可在仓库目录使用 `curl -L 'https://files.grouplens.org/datasets/movielens/ml-1m.zip' -o ml-1m.zip`，然后自行核对官方页面提供的校验值。

   下载后先阅读 [GroupLens 使用条款](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)：研究用途、注明来源，不得未经许可再分发数据；商业用途需另行取得许可。因此本仓库不包含数据集，也不会提供作者本地清洗记录或模型。官方 ZIP 与作者的本地修改版并不相同，不能期待汇报文档里的固定数值。

4. 设置你自己的 `DEEPSEEK_API_KEY` 环境变量。代码不会读取 `OPENAI_API_KEY`，也不会把密钥写入报告或轨迹。`DEEPSEEK_BASE_URL` 默认 `https://api.deepseek.com`，`DEEPSEEK_MODEL` 默认 `deepseek-flash`；两者一般无需配置。请勿把真实密钥放进命令、源码、截图或仓库文件。

5. 确认 `docker info` 成功，然后启动页面：

   ```powershell
   python -m streamlit run streamlit_app.py
   ```

   打开终端显示的本机 URL。全新部署时先提交“请使用默认规则清洗 MovieLens 1M，并评估清洗前后的五维数据质量”；迭代一完成后再提交“请基于已完成的迭代一清洗数据构建迭代二分类、聚类和降维实验”。之后可提问具体 UserID/MovieID。DeepSeek API 需要联网，训练可能需要较长时间。普通迭代二查询会复用已登记模型，但每个自然语言请求仍经过 DeepSeek API。

Windows 用户可在“系统属性 → 高级 → 环境变量”的**用户变量**中新增 `DEEPSEEK_API_KEY`，然后重新打开终端；也可用 PowerShell 设置，但不要把真实密钥写入共享终端历史。Linux/macOS 可由运行环境的秘密管理功能注入同名环境变量。程序优先读取当前进程变量，Windows 下还会读取当前用户环境变量；若两者同时设置，进程变量优先。

以下仅为占位示例，**不要原样运行并提交真实密钥**：

```powershell
[Environment]::SetEnvironmentVariable('DEEPSEEK_API_KEY', '<你的 DeepSeek Key>', 'User')
```

若只想核对是否设置，不显示密钥值，可运行：

```powershell
if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable('DEEPSEEK_API_KEY', 'User'))) { '未设置用户变量' } else { '已设置用户变量' }
```

依赖 DeepSeek 的客户端使用其兼容 OpenAI 的 API 格式；[DeepSeek 官方文档](https://api-docs.deepseek.com/quick_start/agent_integrations/)也说明了默认 URL 与模型名。本项目没有使用 OpenAI API Key。

也可以用命令行触发 Agent：

```powershell
python agent.py --prompt "请清洗 MovieLens 1M 并评估清洗前后的五维质量"
```

正常演示只需在页面提交一次请求，不需要手工逐步运行 Hadoop 命令。

## 真正执行的处理链

1. 将压缩包中的 `ratings.dat`、`users.dat`、`movies.dat` 从 ISO-8859-1 无损转码为 UTF-8，保留无表头、`::` 分隔格式，并记录每个输入文件的 SHA-256。
2. Hadoop Streaming 分别对三张原始表执行质量计数。
3. Hadoop 先清洗用户和电影表，再依据清洗后实体 ID 清洗评分表。Mapper 检查字段、类型、取值及外键；Reducer 对同一逻辑键做去重或冲突隔离。
4. Hadoop 再次对三张清洗后的表执行同一评分程序。
5. 汇总 Hadoop 输出的分子、分母，登记数据版本、规则版本、作业日志、异常处置、时间边界和报告。
6. DeepSeek 根据工具返回的真实报告解释结果，追问时重新读取已完成报告。

Hadoop 采用容器内的本地 MapReduce 模式，仍由真实的 Hadoop Streaming Mapper/Reducer 执行记录处理与指标计数。官方 Hadoop 镜像没有 Python，`Dockerfile.hadoop` 从本地基础镜像组合 Hadoop、Java 和 Python，避免用宿主机脚本冒充 Hadoop 作业。每次完整流程需要 9 个作业。若 Docker、API 或任一作业失败，任务状态记为 `failed`，不会写出完成报告。

## 清洗规则

评分要求合法正整数用户/电影 ID、1–5 整数评分、2000-01-01 至 2003-02-28（UTC）范围内的秒级时间戳，并且外键可关联清洗后的用户和电影表。用户年龄、职业、性别与电影类型均以压缩包中 GroupLens README 给出的编码和词表为准。邮编始终按字符串保存；无法证实其地理真实性。电影标题压缩重复空白，电影类型去掉重复标签；不会猜测缺失年份或纠正未知电影元数据。同一评分事件用 `UserID+MovieID+Timestamp` 识别，不同时间的评分保留。同一逻辑键出现冲突信息时隔离相关记录，不主观选择一条。

对当前提供的压缩包逐行审计后，程序仅增加了可由原行无歧义推断的修复：特定表中的逗号、单冒号或首两个竖线分隔符改回 `::`；末尾字面量 `::EXTRA_FIELD` 去除；评分字面量 `five` 转为 `5`；处于指定历史时间范围的毫秒时间戳转为秒。Hadoop 输出保留每条记录的修复标记，报告分别统计 `repair_attempts` 与最终保留的 `repair_reasons`；修复候选若有同键冲突或无效外键仍须隔离，不计作成功修复。`3.5`、`0`、`6`、空评分、缺字段、未知类型等不会被猜测或补造，进入隔离文件。当前提供的 `ml-1m.zip` 并非未改动的标准 MovieLens 1M 记录集，不应照搬公开数据集的行数或质量结论。

## 五维评分口径

所有分数范围 0–100，原始与清洗后使用同一套规则及同一运行日 UTC 参考时间：

| 维度 | 定义 |
| --- | --- |
| Accurate | 三表通过类型、取值范围和词表校验的记录数 / 三表总记录数；这是可验证代理指标，不等于现实真实性 |
| Complete | 三表所有必填字段均非空的记录数 / 三表总记录数 |
| Unique | 1 - 三表精确重复记录的额外条数 / 三表总记录数 |
| Up-to-date | 近 365 天内的合法评分条数 / ratings 总条数；用户和电影没有更新时间，不纳入 |
| Consistent | 三表规则合法且评分外键可关联的记录数 / 三表总记录数 |

数据集是历史基准集，时效性很可能长期为零；清洗不会使旧评分变新。隔离和去重缩小分母，因此分数变化必须结合数据量、处置原因和隔离文件解释。系统不会为了抬高分数改变评分规则。

## 产物和后续迭代

每次任务生成 `artifacts/runs/<task_id>/`，其中含 `raw/`、`clean/`、`quarantine/`、`hadoop/`、`logs/`、`status.json`、`report.json`、`report.md`、`agent_traces/`。每次 Agent 交互另存一份轨迹，追问不会覆盖初次运行记录。清洗数据版本为三表 SHA-256 的组合哈希；后续迭代应以该版本和规则版本校验输入。可执行 `python validate_run.py <task_id>` 独立核对任务状态、九个作业日志、记录守恒、输出行数、文件哈希、数据版本与评分汇总。

`T1/T2` 从清洗后评分时间戳的 80%/90% 分位确定，时间戳相同时归入较早分区；报告保存三个分区的实际数量。后续训练仅能使用 `T1` 及之前的信息，验证期为 `T1<Timestamp≤T2`，测试期为 `Timestamp>T2`。

MovieLens 数据随附的 README 规定研究用途和再分发限制。生成的清洗数据仅保存在本机 `artifacts/`，提交或共享前请先核对原始许可。

## 迭代二：分类、聚类、降维

迭代二从**已完成且哈希校验通过**的迭代一清洗产物读取数据；缺少前置报告/文件、版本不一致或 T1/T2 无效时立即报“数据不可用”，不会用示例数据代替。安装更新后的 `requirements.txt` 后可在页面提交自然语言任务，也可先直接生成实验：

```powershell
python -m pip install -r requirements.txt
python -X utf8 iteration2.py
python -m streamlit run streamlit_app.py
```

页面可查看两种分类器验证指标、入选模型的测试指标、用户群体摘要、二维坐标图、降维前后轮廓系数、实际运行成本和结果文件。页面的自然语言请求与追问由 `agent.py` 调用 **DeepSeek API**；DeepSeek 选择并真正调用工具，底层数值由本地训练模型和登记产物产生。离线命令 `iteration2.py` 用于可复现地建立模型，不会冒充 LLM Agent。运行 `python -m pytest -q` 可检查时间防泄漏、模型查询与产物。

六个独立的 Agent 算法工具为 `classify_logistic_sgd`、`classify_decision_tree`、`cluster_kmeans`、`cluster_dbscan`、`reduce_pca`、`reduce_nmf`。另外有完整实验构建和报告读取工具。分类课件第 24 页明确列出 Logistic Regression 与 Decision Tree；用户未提供聚类/降维课件，因此 KMeans、DBSCAN、PCA、NMF 按迭代二需求选型，**不声称来自未提供的 PPT**。

分类目标是一次评分是否至少 4 分。训练行的用户与电影历史统计只使用严格早于该行时间戳的事件；同一秒的事件互不可见。验证、测试与查询统一冻结在 T1 的训练历史快照，不使用未来评分特征。静态用户/电影目录元数据视为训练时可用，但数据本身没有元数据发布时间，报告明确记为限制。验证集按 Average Precision 选择模型，仅入选模型在测试集评估。模型文件、静态与历史特征快照均保存在本地。

用户画像只来自 T1 前评分，训练期不足 20 条的用户不分群。画像含 18 类电影的曝光占比、18 类的高分率及用户平均分/历史量，共 38 维。KMeans 在训练期轮廓系数上选择 k=4/6/8；DBSCAN 基于第 10 近邻距离分位候选选择 `eps`，输出噪声标记。群体摘要按真实训练评分计算，不对噪声用户强行解释。PCA 和 NMF 均保留默认 2/8 维模型；可由 Agent 请求 2–16 维、信息保留阈值、指定聚类算法及有界参数，额外模型另存版本化产物。PCA 的解释方差比与 NMF 的重构方差代理**口径不同**，模型重建 RMSE 也处于不同输入空间，不能直接比较绝对数值。

默认产物位于 `artifacts/iteration2/<迭代一任务ID>/iteration2-v2/`，包括报告和状态文件、分类模型与特征快照、用户画像、聚类模型与成员表、PCA/NMF 模型与坐标、二维前后聚类成员表、Agent API 工具调用轨迹；自定义参数的额外模型在 `variants/`。`report.json` 记录源数据版本、时间边界、真实参数与指标、文件引用和局限，供第三轮使用。原始清洗文件与模型仅在本机，不应未经许可再分发 MovieLens 数据。

可直接演示的自然语言请求：

1. `请基于迭代一清洗数据构建完整迭代二实验，比较分类、聚类和降维各两种算法。`
2. `请用两种分类算法预测用户659对电影1是否会打至少4分，列出概率、验证指标和模型版本。`
3. `请用KMeans与DBSCAN比较用户659的群体，说明群体规模、代表电影与噪声。`
4. `请用PCA和NMF展示用户659的二维坐标、信息保留和KMeans聚类前后轮廓系数。`
5. `请用KMeans的3个群体重新分析用户659，并用PCA压缩到3维，对照压缩前后聚类。`

演示时不要把一次评分事件的高分概率说成“用户一定会观看/喜欢”，不要把轮廓系数当作真实群体准确率。DBSCAN 在当前数据上的高轮廓系数伴随极不平衡的簇，应展示群体大小后再解释；测试集没有用于聚类或降维选型。

## 测试与公开仓库边界

开发者可安装 `requirements-dev.txt` 并运行 `python -m pytest -q`。公开仓库的 CI 在**没有**数据、模型、Docker 和 API 密钥的全新检出上运行单元测试；依赖本地真实产物的集成测试会跳过。要验证完整端到端流程，仍需按“从新电脑部署”准备官方数据、Docker 和自己的 DeepSeek Key，然后运行迭代一、迭代二并检查页面与 `artifacts/` 中的报告。

`.gitignore` 排除了 `ml-1m.zip`、`artifacts/`、`.env*` 和 `.streamlit/secrets.toml`。不要强制添加这些文件。课程要求文档、[Agent 演示说明](Agent演示说明.md)、两份汇报辅助文档和源代码可以公开；历史演示结果不是承诺其他输入会得到相同数字。

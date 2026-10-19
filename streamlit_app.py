"""Interactive front end for the DeepSeek MovieLens governance and analysis agent."""
import pandas as pd
import streamlit as st

from agent import handle, latest_report
from iteration2 import latest_analysis
from pipeline import RUNS, ZIP_PATH

st.set_page_config(page_title="MovieLens 分析 Agent", page_icon=":material/movie:", layout="wide")
st.title("MovieLens 1M 分析 Agent")
st.caption("自然语言 → DeepSeek 真实工具调用 → 清洗、分类、聚类与降维 → 可追溯证据")
if not ZIP_PATH.is_file():
    st.warning("尚未找到 ml-1m.zip。请先按 README 从 GroupLens 获取数据，放到项目根目录；本仓库不分发原始数据。")

has_source_report = any(RUNS.glob("*/report.json")) if RUNS.exists() else False
default_request = (
    "请基于已完成的迭代一清洗数据，构建迭代二分类、聚类和降维实验，并对比各两种算法的真实评价结果及局限。"
    if has_source_report else
    "请使用默认规则清洗 MovieLens 1M，并评估清洗前后的五维数据质量。"
)


@st.cache_data(ttl="10m", max_entries=12)
def read_csv(path: str, modified_ns: int) -> pd.DataFrame:
    return pd.read_csv(path)

st.session_state.setdefault("result", None)
st.session_state.setdefault("conversation", [])
st.session_state.setdefault("last_error", None)
st.session_state.setdefault("last_stage", None)

with st.form("request_form", border=True):
    request = st.text_area(
        "任务请求",
        value=default_request,
        height=105,
        help="请求由 DeepSeek API 判断并选择工具；模型、坐标与指标由本地真实计算产生。",
    )
    submitted = st.form_submit_button("运行 Agent", type="primary", icon=":material/play_arrow:")

if submitted:
    st.session_state.last_error = None
    st.session_state.last_stage = "调用 DeepSeek API"
    with st.status("准备调用 DeepSeek", expanded=True) as status:
        try:
            def progress(stage: str) -> None:
                st.session_state.last_stage = stage
                status.update(label=f"正在执行：{stage}", state="running")
                st.write(stage)

            result = handle(request, progress)
            st.session_state.result = result
            st.session_state.conversation = [
                {"role": "user", "content": request},
                {"role": "assistant", "content": result["answer"]},
            ]
            status.update(label="任务完成", state="complete", expanded=False)
        except Exception as exc:
            st.session_state.last_error = str(exc)
            status.update(label=f"任务失败：{st.session_state.last_stage}", state="error", expanded=True)
            st.error(str(exc))

if st.session_state.last_error:
    st.error(f"最近一次任务失败 · 阶段：{st.session_state.last_stage} · 原因：{st.session_state.last_error}")

result = st.session_state.result
report = result["report"] if result else None
if report is None:
    try:
        report = latest_report()
        st.info("正在展示上一次完成的真实报告。")
    except FileNotFoundError:
        st.info("尚无完成的评估报告。提交请求后会显示真实运行结果。")

if report:
    run_dir = RUNS / report["task_id"]
    st.subheader("任务与数据版本")
    with st.container(horizontal=True):
        st.metric("任务状态", report["status"], border=True)
        st.metric("Hadoop 作业", len(report["hadoop_jobs"]), border=True)
        st.metric("清洗数据版本", report["data_version"][:12], border=True)
    st.caption(
        f"任务 ID：{report['task_id']} · 规则：{report['rule_version']} · "
        f"评分：{report['score_version']} · T1：{report['time_split']['T1_utc']} · "
        f"T2：{report['time_split']['T2_utc']}"
    )

    st.subheader("五维质量评分")
    before, after = report["scores"]["before"], report["scores"]["after"]
    scores = pd.DataFrame([
        {"维度": name, "清洗前": before[name], "清洗后": after[name],
         "变化（百分点）": round(after[name] - before[name], 4)}
        for name in before
    ])
    st.dataframe(scores, hide_index=True)
    st.bar_chart(scores.set_index("维度")[["清洗前", "清洗后"]], y_label="分数（0–100）")
    st.caption("准确性为规则可验证的代理评分，不能证明现实真实性；时效性以运行日近 365 天的评分占比计算。")

    st.subheader("数据处置")
    rows = []
    for table in ("users", "movies", "ratings"):
        actions = report["cleaning"][table]["actions"]
        rows.append({
            "数据表": table, "原始记录": report["counts"]["before"][table]["total"],
            "保留": actions.get("keep", 0),
            "规范化": report["cleaning"][table]["reasons"].get("normalized", 0),
            "去重": actions.get("deduplicate", 0),
            "隔离": actions.get("quarantine", 0),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("去重和隔离会缩小评分分母；隔离记录并未修复原始数据。")

    with st.expander("异常证据与 Hadoop 作业日志"):
        for table in ("users", "movies", "ratings"):
            st.markdown(f"**{table}**")
            st.write("处置原因")
            st.json(report["cleaning"][table]["reasons"])
            st.write("检测到的无歧义修复尝试")
            st.json(report["cleaning"][table].get("repair_attempts", {}))
            st.write("通过冲突和外键检查后保留的修复")
            st.json(report["cleaning"][table].get("repair_reasons", {}))
            examples = report["cleaning"][table]["examples"]
            if examples:
                st.dataframe(pd.DataFrame(examples), hide_index=True)
        st.dataframe(pd.DataFrame(report["hadoop_jobs"]), hide_index=True)

    with st.expander("清洗后样例与结果文件"):
        table = st.selectbox("数据表", ("ratings", "users", "movies"))
        sample_path = run_dir / "clean" / f"{table}.dat"
        with sample_path.open(encoding="utf-8") as source:
            sample = [next(source, "").rstrip("\n") for _ in range(10)]
        st.code("\n".join(x for x in sample if x), language="text")
        report_path = run_dir / "report.md"
        st.download_button("下载评估报告", report_path.read_bytes(),
                           file_name=f"{report['task_id']}-report.md",
                           mime="text/markdown", icon=":material/download:")
        st.caption(f"清洗数据、隔离记录和作业日志保存在 {run_dir}")

st.subheader("迭代二 · 分类、聚类与降维")
analysis = result.get("analysis_report") if result else None
analysis_dir = None
if analysis is None:
    try:
        analysis, analysis_dir = latest_analysis()
    except (FileNotFoundError, ValueError) as exc:
        st.info(str(exc))
else:
    try:
        _, analysis_dir = latest_analysis()
    except (FileNotFoundError, ValueError):
        analysis_dir = None

if analysis and analysis_dir:
    with st.container(horizontal=True):
        st.metric("状态", analysis["status"], border=True)
        st.metric("训练期用户画像", analysis["clustering"]["eligible_users"], border=True)
        st.metric("入选分类器", analysis["classification"]["selected"], border=True)
    st.caption(
        f"数据版本 {analysis['data_version'][:16]} · 来源任务 {analysis['source_task_id']} · "
        f"T1={analysis['T1']} · T2={analysis['T2']} · "
        f"训练/验证/测试={analysis['split_counts']['train']:,}/{analysis['split_counts']['validation']:,}/{analysis['split_counts']['test']:,}"
    )
    st.info("分类算法 Logistic Regression 与 Decision Tree 已在提供的分类课件中核对；聚类和降维算法由任务需求选型，未提供对应课件。")

    st.markdown("#### 分类实验")
    classifier_rows = []
    for name, item in analysis["classification"]["algorithms"].items():
        classifier_rows.append({
            "算法": name, "验证 AP": item["validation"]["average_precision"],
            "验证 ROC-AUC": item["validation"]["roc_auc"],
            "验证 F1@0.5": item["validation"]["f1_at_0_5"],
            "测试 AP": item.get("test", {}).get("average_precision") if isinstance(item.get("test"), dict) else None,
            "模型版本": item["version"],
        })
    st.dataframe(pd.DataFrame(classifier_rows), hide_index=True)
    st.caption("预测目标是评分事件中是否给出至少 4 分；验证集 AP 选型后，只对入选分类器计算测试指标。未入选模型的测试指标不是 0，而是未评估。")

    st.markdown("#### 用户分群")
    clustering_rows = []
    for name, item in analysis["clustering"]["algorithms"].items():
        clustering_rows.append({"算法": name, "训练期轮廓系数": item["silhouette_train"],
                                "群体数": item["cluster_count"], "噪声用户": item["noise_count"],
                                "运行秒数": item["fit_seconds"], "版本": item["version"]})
    st.dataframe(pd.DataFrame(clustering_rows), hide_index=True)
    st.caption(
        f"仅对训练期至少 20 条评分的用户分群；有 1–19 条评分的用户 {analysis['clustering']['insufficient_history_users']} 人，"
        f"训练期 0 条评分的用户 {analysis['clustering'].get('no_training_history_users', '未登记')} 人。"
        "DBSCAN 的 -1 是噪声，不是可解释的兴趣群体；轮廓系数不是外部真值准确率。"
    )

    st.markdown("#### 降维与聚类对照")
    reduction_rows = []
    for name, variants in analysis["reduction"]["algorithms"].items():
        for dims, item in variants.items():
            for clustering_name, comparison in item["cluster_comparison"].items():
                reduction_rows.append({"降维": name, "维数": int(dims), "信息保留": item["retention"],
                                       "保留指标": item["retention_metric"], "聚类": clustering_name,
                                       "原画像轮廓系数": analysis["clustering"]["algorithms"][clustering_name]["silhouette_train"],
                                       "降维后轮廓系数": comparison["silhouette_reduced"],
                                       "模型大小 KB": round(item["model_bytes"] / 1024, 1),
                                       "坐标大小 KB": round(item["coordinates_bytes"] / 1024, 1),
                                       "耗时秒": item["fit_and_comparison_seconds"]})
    st.dataframe(pd.DataFrame(reduction_rows), hide_index=True)
    st.caption("PCA 使用解释方差比，NMF 使用重构方差代理；两者定义不同。轮廓系数在不同空间可能变动，不代表真实群体标签正确率提高。")
    with st.container(border=True):
        method = st.segmented_control("二维表示", ["pca", "nmf"], default="pca", key="plot_method")
        if method:
            coordinate = analysis["reduction"]["algorithms"][method]["2"]
            coord_path = analysis_dir / coordinate["coordinate_file"]
            assignment_path = analysis_dir / analysis["clustering"]["algorithms"]["kmeans"]["assignment_file"]
            plot = read_csv(str(coord_path), coord_path.stat().st_mtime_ns).merge(
                read_csv(str(assignment_path), assignment_path.stat().st_mtime_ns)[["UserID", "cluster"]], on="UserID")
            plot["群体"] = plot["cluster"].astype(str)
            st.scatter_chart(plot, x="component_1", y="component_2", color="群体",
                             x_label="第 1 分量", y_label="第 2 分量")
            st.caption("颜色是原始画像 KMeans 群体；坐标由当前降维模型实际生成。")

    with st.expander("群体摘要、结果样例和下载"):
        for name, item in analysis["clustering"]["algorithms"].items():
            st.markdown(f"**{name}**")
            summary_rows = []
            for group_id, summary in item["summaries"].items():
                summary_rows.append({"群体": group_id, "人数": summary["users"],
                                     "高分率较高类型": " / ".join(summary["top_genres_by_high_rating_rate"]),
                                     "代表电影": " / ".join(summary["representative_movies_by_high_rating_count"]),
                                     "训练期平均分": summary["mean_training_rating"]})
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True)
            assignment_path = analysis_dir / item["assignment_file"]
            st.dataframe(read_csv(str(assignment_path), assignment_path.stat().st_mtime_ns).head(10), hide_index=True)
            st.download_button(f"下载 {name} 用户群体", assignment_path.read_bytes(),
                               file_name=item["assignment_file"], mime="text/csv", key=f"download_{name}")
        for method in ("pca", "nmf"):
            coord_path = analysis_dir / analysis["reduction"]["algorithms"][method]["2"]["coordinate_file"]
            st.download_button(f"下载 {method} 二维坐标", coord_path.read_bytes(),
                               file_name=coord_path.name, mime="text/csv", key=f"download_{method}")
        st.download_button("下载完整迭代二评价报告", (analysis_dir / "report.json").read_bytes(),
                           file_name=f"{analysis['source_task_id']}-iteration2-report.json",
                           mime="application/json")
    with st.expander("证据、参数与局限"):
        st.write(analysis["algorithm_selection_basis"])
        for note in analysis["limitations"]:
            st.write("- " + note)
        st.json({"classification": {k: v["parameters"] for k, v in analysis["classification"]["algorithms"].items()},
                 "clustering": {k: v["parameters"] for k, v in analysis["clustering"]["algorithms"].items()},
                 "reduction": {k: {d: v["parameters"] for d, v in vals.items()}
                               for k, vals in analysis["reduction"]["algorithms"].items()}})

if result and result.get("tool_results"):
    st.subheader("本次 Agent 工具证据")
    for item in result["tool_results"]:
        with st.expander(item["name"], expanded=item["name"].startswith(("classify_", "cluster_", "reduce_"))):
            st.json(item["result"])

st.subheader("Agent 解释与追问")
for message in st.session_state.conversation:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("追问评分、群体、二维坐标或清洗依据；例如：比较用户 1 对电影 1 的两种分类预测", submit_mode="disable"):
    st.session_state.conversation.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            followup = handle(question)
            st.write(followup["answer"])
            st.session_state.conversation.append({"role": "assistant", "content": followup["answer"]})
            if followup["report"]:
                st.session_state.result = followup
        except Exception as exc:
            st.error(str(exc))

"""Leakage-aware, versioned MovieLens classification / clustering / reduction."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, roc_auc_score, silhouette_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from pipeline import ROOT, RUNS, atomic_json, digest

OUT = ROOT / "artifacts" / "iteration2"
VERSION = "iteration2-v2"
GENRES = ("Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
          "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western")
CLASSIFIERS = ("logistic_sgd", "decision_tree")
CLUSTERERS = ("kmeans", "dbscan")
REDUCERS = ("pca", "nmf")
MIN_HISTORY = 20


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def source_report(task_id: str | None = None) -> tuple[dict, Path]:
    if task_id is not None and not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}", task_id):
        raise ValueError("输入无效：迭代一任务 ID 格式不合法")
    reports = [RUNS / task_id / "report.json"] if task_id else sorted(RUNS.glob("*/report.json"), reverse=True)
    if not reports:
        raise FileNotFoundError("数据不可用：没有完成的迭代一报告")
    path = reports[0]
    if not path.is_file():
        raise FileNotFoundError(f"数据不可用：报告不存在 {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "completed":
        raise ValueError("数据不可用：迭代一未完成")
    folder = path.parent / "clean"
    files = [folder / f"{name}.dat" for name in ("users", "movies", "ratings")]
    if any(not p.is_file() for p in files):
        raise FileNotFoundError("数据不可用：缺少迭代一清洗文件")
    for name, file in zip(("users", "movies", "ratings"), files):
        expected = report["cleaning"][name]["clean_sha256"]
        if digest(file) != expected:
            raise ValueError(f"数据不可用：{name} 清洗文件哈希与报告不符")
    version = hashlib.sha256("".join(digest(p) for p in files).encode()).hexdigest()
    if version != report["data_version"]:
        raise ValueError("数据不可用：清洗数据版本不一致")
    split = report.get("time_split", {})
    if not isinstance(split.get("T1"), int) or not isinstance(split.get("T2"), int) or split["T1"] >= split["T2"]:
        raise ValueError("数据不可用：T1/T2 无效")
    return report, folder


def output_dir(report: dict) -> Path:
    return OUT / report["task_id"] / VERSION


def _verify_analysis_files(report: dict, folder: Path) -> None:
    required = [report["classification"]["feature_file"], report["clustering"]["profile_file"]]
    checksums = {report["classification"]["feature_file"]: report["classification"]["feature_sha256"],
                 report["clustering"]["profile_file"]: report["clustering"]["profile_sha256"]}
    for item in report["classification"]["algorithms"].values():
        required.append(item["model_file"])
        checksums[item["model_file"]] = item["model_sha256"]
    for item in report["clustering"]["algorithms"].values():
        required.extend((item["model_file"], item["assignment_file"]))
        checksums[item["assignment_file"]] = item["assignment_sha256"]
    for variants in report["reduction"]["algorithms"].values():
        for item in variants.values():
            required.extend((item["model_file"], item["coordinate_file"]))
            checksums[item["coordinate_file"]] = item["coordinate_sha256"]
            required.extend(c["assignment_file"] for c in item["cluster_comparison"].values())
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError("未完成：迭代二结果文件缺失：" + ", ".join(missing[:5]))
    invalid = [name for name, sha in checksums.items() if digest(folder / name) != sha]
    if invalid:
        raise ValueError("数据不可用：迭代二结果文件哈希不符：" + ", ".join(invalid[:5]))


def latest_analysis() -> tuple[dict, Path]:
    source, _ = source_report()
    path = output_dir(source) / "report.json"
    if not path.is_file():
        raise FileNotFoundError("未完成：当前迭代一数据版本尚无迭代二分析结果")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("data_version") != source["data_version"] or report.get("source_task_id") != source["task_id"]:
        raise ValueError("数据不可用：迭代二结果与当前迭代一版本不一致")
    _verify_analysis_files(report, path.parent)
    return report, path.parent


def _read_tables(folder: Path, split: dict):
    users = pd.read_csv(folder / "users.dat", sep="::", engine="python", header=None,
                        names=["UserID", "Gender", "Age", "Occupation", "Zip"])
    movies = pd.read_csv(folder / "movies.dat", sep="::", engine="python", header=None,
                         names=["MovieID", "Title", "Genres"], encoding="utf-8")
    ratings = pd.read_csv(folder / "ratings.dat", sep="::", engine="python", header=None,
                          names=["UserID", "MovieID", "Rating", "Timestamp"], dtype=np.int64)
    t1, t2 = split["T1"], split["T2"]
    train = ratings.loc[ratings.Timestamp <= t1].sort_values(["Timestamp", "UserID", "MovieID"]).reset_index(drop=True)
    validation = ratings.loc[(ratings.Timestamp > t1) & (ratings.Timestamp <= t2)].reset_index(drop=True)
    test = ratings.loc[ratings.Timestamp > t2].reset_index(drop=True)
    counts = {"train": len(train), "validation": len(validation), "test": len(test)}
    if counts != split["counts"]:
        raise ValueError(f"数据不可用：时间切分计数与迭代一不一致 {counts}")
    if not len(train) or not len(validation) or not len(test):
        raise ValueError("数据不可用：训练、验证或测试分区为空")
    return users, movies, train, validation, test


def _prior_at_event(train: pd.DataFrame, id_col: str) -> np.ndarray:
    grouped = train.assign(high=(train.Rating >= 4).astype(np.int32)).groupby(
        [id_col, "Timestamp"], sort=True).agg(n=("Rating", "size"), total=("Rating", "sum"), high=("high", "sum"))
    for col in ("n", "total", "high"):
        grouped["before_" + col] = grouped.groupby(level=0)[col].cumsum() - grouped[col]
    joined = train[[id_col, "Timestamp"]].merge(grouped[["before_n", "before_total", "before_high"]].reset_index(),
                                                  on=[id_col, "Timestamp"], how="left", validate="many_to_one")
    return joined[["before_n", "before_total", "before_high"]].to_numpy(dtype=np.float32)


def _static_arrays(users: pd.DataFrame, movies: pd.DataFrame):
    max_user, max_movie = int(users.UserID.max()), int(movies.MovieID.max())
    u = np.zeros((max_user + 1, 1 + 7 + 21), np.float32)
    m = np.zeros((max_movie + 1, 1 + len(GENRES)), np.float32)
    valid_u = np.zeros(max_user + 1, dtype=bool)
    valid_m = np.zeros(max_movie + 1, dtype=bool)
    age_codes = {age: i for i, age in enumerate(sorted(users.Age.unique()))}
    genre_index = {g: i for i, g in enumerate(GENRES)}
    for row in users.itertuples():
        uid = int(row.UserID)
        valid_u[uid] = True
        u[uid, 0] = float(row.Gender == "F")
        u[uid, 1 + age_codes[row.Age]] = 1
        u[uid, 8 + int(row.Occupation)] = 1
    for row in movies.itertuples():
        mid = int(row.MovieID)
        valid_m[mid] = True
        year = str(row.Title).rpartition("(")[2].rstrip(")")
        m[mid, 0] = (int(year) - 1900) / 110 if year.isdigit() else 0
        for genre in str(row.Genres).split("|"):
            if genre in genre_index:
                m[mid, 1 + genre_index[genre]] = 1
    return u, m, valid_u, valid_m


def _history_features(prior: np.ndarray) -> np.ndarray:
    count, total, high = prior.T
    return np.stack((np.log1p(count) / 10, np.divide(total, count * 5, out=np.zeros_like(total), where=count > 0),
                     np.divide(high, count, out=np.zeros_like(high), where=count > 0)), axis=1).astype(np.float32)


def _training_features(train, u, m):
    up = _prior_at_event(train, "UserID")
    mp = _prior_at_event(train, "MovieID")
    return np.hstack((u[train.UserID.to_numpy()], m[train.MovieID.to_numpy()],
                      _history_features(up), _history_features(mp))).astype(np.float32)


def _snapshots(train, max_user, max_movie):
    user = np.zeros((max_user + 1, 3), np.float32)
    movie = np.zeros((max_movie + 1, 3), np.float32)
    data = train.assign(high=(train.Rating >= 4).astype(np.int32))
    for col, array in (("UserID", user), ("MovieID", movie)):
        sums = data.groupby(col).agg(n=("Rating", "size"), total=("Rating", "sum"), high=("high", "sum"))
        ids = sums.index.to_numpy(dtype=np.int64)
        array[ids] = sums[["n", "total", "high"]].to_numpy(np.float32)
    return user, movie


def _future_features(rows, u, m, up, mp):
    return np.hstack((u[rows.UserID.to_numpy()], m[rows.MovieID.to_numpy()],
                      _history_features(up[rows.UserID.to_numpy()]),
                      _history_features(mp[rows.MovieID.to_numpy()]))).astype(np.float32)


def _classification_metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    return {"rows": int(len(y)), "positive_rate": round(float(np.mean(y)), 6),
            "roc_auc": round(float(roc_auc_score(y, prob)), 6),
            "average_precision": round(float(average_precision_score(y, prob)), 6),
            "f1_at_0_5": round(float(f1_score(y, pred, zero_division=0)), 6),
            "balanced_accuracy_at_0_5": round(float(balanced_accuracy_score(y, pred)), 6)}


def _classify(train, val, test, u, m, out, progress):
    progress("分类：构造严格早于评分事件的历史特征")
    x_train = _training_features(train, u, m)
    up, mp = _snapshots(train, len(u) - 1, len(m) - 1)
    x_val = _future_features(val, u, m, up, mp)
    y_train = (train.Rating.to_numpy() >= 4).astype(np.int8)
    y_val = (val.Rating.to_numpy() >= 4).astype(np.int8)
    models = {}
    configs = {
        "logistic_sgd": {"loss": "log_loss", "alpha": 0.0001, "max_iter": 60, "tol": 0.0001, "random_state": 42},
        "decision_tree": {"max_depth": 12, "min_samples_leaf": 100, "random_state": 42},
    }
    for name in CLASSIFIERS:
        progress("分类：训练 " + name)
        start = time.perf_counter()
        if name == "logistic_sgd":
            scaler = StandardScaler()
            fit_x = scaler.fit_transform(x_train)
            estimator = SGDClassifier(**configs[name]).fit(fit_x, y_train)
            val_prob = estimator.predict_proba(scaler.transform(x_val))[:, 1]
        else:
            scaler = None
            estimator = DecisionTreeClassifier(**configs[name]).fit(x_train, y_train)
            val_prob = estimator.predict_proba(x_val)[:, 1]
        model_path = out / f"classification-{name}.joblib"
        joblib.dump({"estimator": estimator, "scaler": scaler}, model_path)
        models[name] = {"version": f"{VERSION}-{name}", "algorithm": type(estimator).__name__,
                        "parameters": configs[name], "validation": _classification_metrics(y_val, val_prob),
                        "fit_and_validation_seconds": round(time.perf_counter() - start, 3),
                        "model_bytes": model_path.stat().st_size, "model_sha256": digest(model_path),
                        "model_file": model_path.name}
    selected = max(CLASSIFIERS, key=lambda n: models[n]["validation"]["average_precision"])
    progress("分类：验证集选型完成后，仅评估入选模型的测试集")
    winner = joblib.load(out / models[selected]["model_file"])
    x_test = _future_features(test, u, m, up, mp)
    if winner["scaler"] is not None:
        x_test = winner["scaler"].transform(x_test)
    models[selected]["test"] = _classification_metrics((test.Rating.to_numpy() >= 4).astype(np.int8),
                                                         winner["estimator"].predict_proba(x_test)[:, 1])
    feature_path = out / "classification-features.joblib"
    joblib.dump({"user_static": u, "movie_static": m, "user_history_T1": up,
                 "movie_history_T1": mp}, feature_path)
    return {"target": "Rating>=4", "feature_policy": "训练行仅用同一实体严格早于该行 Timestamp 的历史；验证、测试和在线预测均冻结在 T1 的历史快照；静态用户/电影元数据来自清洗表",
            "selection_metric": "validation average_precision", "selected": selected,
            "algorithms": models, "feature_file": feature_path.name, "feature_sha256": digest(feature_path),
            "train_positive_rate": round(float(np.mean(y_train)), 6)}


def _profiles(train, movies, progress):
    progress("聚类：只用训练期评分构造用户类型偏好")
    movie_genres = {}
    for row in movies.itertuples():
        movie_genres[int(row.MovieID)] = [i for i, g in enumerate(GENRES) if g in str(row.Genres).split("|")]
    users = np.sort(train.UserID.unique())
    index = {int(uid): i for i, uid in enumerate(users)}
    exposures = np.zeros((len(users), len(GENRES)), np.float32)
    liked = np.zeros_like(exposures)
    for uid, mid, rating in train[["UserID", "MovieID", "Rating"]].itertuples(index=False, name=None):
        ix = index[int(uid)]
        for genre in movie_genres.get(int(mid), ()):
            exposures[ix, genre] += 1
            liked[ix, genre] += int(rating >= 4)
    aggregates = train.groupby("UserID").Rating.agg(["size", "mean"])
    counts = aggregates.loc[users, "size"].to_numpy()
    valid = counts >= MIN_HISTORY
    if int(valid.sum()) < 20:
        raise ValueError("数据不可用：训练期历史达到最小条数的用户不足 20")
    count_features = np.divide(exposures, counts[:, None], out=np.zeros_like(exposures), where=counts[:, None] > 0)
    like_features = np.divide(liked, exposures, out=np.zeros_like(liked), where=exposures > 0)
    mean = aggregates.loc[users, "mean"].to_numpy(np.float32)[:, None] / 5
    log_count = (np.log1p(counts) / 10).astype(np.float32)[:, None]
    matrix = np.hstack((count_features, like_features, mean, log_count))[valid]
    return users[valid], counts[valid], matrix, exposures[valid], liked[valid], movie_genres


def _silhouette(x, labels):
    non_noise = labels >= 0
    labs = labels[non_noise]
    if len(set(labs)) < 2 or len(labs) < 3:
        return None
    value = silhouette_score(x[non_noise], labs, sample_size=min(1000, int(non_noise.sum())), random_state=42)
    return round(float(value), 6)


def _cluster_fit(x, algorithm, progress):
    if algorithm == "kmeans":
        candidates = []
        for k in (4, 6, 8):
            start = time.perf_counter()
            model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(x)
            score = _silhouette(x, model.labels_)
            candidates.append((score, k, model, time.perf_counter() - start))
        score, k, model, seconds = max(candidates, key=lambda item: item[0] if item[0] is not None else -2)
        return model, model.labels_, {"n_clusters": k, "n_init": 10, "random_state": 42,
                                     "candidate_silhouette": {str(item[1]): item[0] for item in candidates}}, seconds
    if algorithm == "dbscan":
        min_samples = 10
        neighbors = NearestNeighbors(n_neighbors=min_samples).fit(x)
        distances = neighbors.kneighbors(x)[0][:, -1]
        candidates = []
        for quantile in (0.65, 0.75, 0.85, 0.95):
            eps = float(np.quantile(distances, quantile))
            start = time.perf_counter()
            model = DBSCAN(eps=eps, min_samples=min_samples).fit(x)
            score = _silhouette(x, model.labels_)
            noise_fraction = float(np.mean(model.labels_ == -1))
            candidates.append((score, noise_fraction, quantile, eps, model, time.perf_counter() - start))
        eligible = [c for c in candidates if c[0] is not None and c[1] <= 0.5]
        selected = max(eligible or candidates, key=lambda c: c[0] if c[0] is not None else -2)
        score, noise, quantile, eps, model, seconds = selected
        return model, model.labels_, {"eps": round(eps, 6), "min_samples": min_samples,
                                     "k_distance_quantile": quantile,
                                     "candidate_silhouette": {str(c[2]): c[0] for c in candidates}}, seconds
    raise ValueError("未知聚类算法")


def _cluster_summary(labels, user_ids, train, movies, exposure, liked):
    summaries = {}
    movie_titles = dict(zip(movies.MovieID.astype(int), movies.Title.astype(str)))
    user_to_label = dict(zip(user_ids.astype(int), labels.astype(int)))
    focused = train.loc[train.UserID.isin(user_to_label)].copy()
    focused["group"] = focused.UserID.map(user_to_label)
    for label in sorted(set(labels)):
        if label == -1:
            continue  # DBSCAN noise does not define a coherent taste group.
        positions = labels == label
        exposure_total = exposure[positions].sum(axis=0)
        preference = np.divide(liked[positions].sum(axis=0), exposure_total,
                               out=np.zeros(len(GENRES), np.float32), where=exposure_total > 0)
        supported = np.flatnonzero(exposure_total >= max(10, int(positions.sum())))
        top_genres = [GENRES[i] for i in sorted(supported, key=lambda ix: preference[ix], reverse=True)[:3]]
        subset = focused[(focused.group == label) & (focused.Rating >= 4)]
        top_movies = subset.groupby("MovieID").size().sort_values(ascending=False).head(3)
        summaries[str(int(label))] = {"users": int(positions.sum()), "top_genres_by_high_rating_rate": top_genres,
                                      "representative_movies_by_high_rating_count": [movie_titles.get(int(mid), str(mid)) for mid in top_movies.index],
                                      "mean_training_rating": round(float(focused.loc[focused.group == label, "Rating"].mean()), 4)}
    return summaries


def _clusters(x, ids, counts, train, movies, exposure, liked, all_user_count, out, progress):
    scaler = StandardScaler()
    standardized = scaler.fit_transform(x)
    saved = out / "user-profiles.joblib"
    joblib.dump({"user_ids": ids, "history_counts": counts, "matrix": x, "scaled": standardized,
                 "scaler": scaler}, saved)
    algorithms = {}
    for name in CLUSTERERS:
        progress("聚类：拟合 " + name)
        model, labels, parameters, seconds = _cluster_fit(standardized, name, progress)
        model_file = out / f"cluster-{name}.joblib"
        joblib.dump(model, model_file)
        assignment = out / f"cluster-{name}.csv"
        pd.DataFrame({"UserID": ids, "cluster": labels, "noise": labels == -1,
                      "training_ratings": counts}).to_csv(assignment, index=False)
        algorithms[name] = {"version": f"{VERSION}-{name}", "algorithm": type(model).__name__,
                            "parameters": parameters, "silhouette_train": _silhouette(standardized, labels),
                            "cluster_count": int(len(set(labels) - {-1})), "noise_count": int(np.sum(labels == -1)),
                            "cluster_sizes": {str(int(label)): int(n) for label, n in Counter(labels).items()},
                            "fit_seconds": round(seconds, 3), "model_bytes": model_file.stat().st_size,
                            "model_file": model_file.name, "assignment_file": assignment.name,
                            "assignment_sha256": digest(assignment),
                            "summaries": _cluster_summary(labels, ids, train, movies, exposure, liked)}
    return {"profile_policy": f"仅使用 Timestamp<=T1 的评分；至少 {MIN_HISTORY} 条历史评分的用户可分群；36 个类型曝光/高分率特征和平均分、历史数",
            "eligible_users": int(len(ids)), "insufficient_history_users": int(train.UserID.nunique() - len(ids)),
            "no_training_history_users": int(all_user_count - train.UserID.nunique()),
            "profile_dimensions": int(x.shape[1]), "profile_file": saved.name, "profile_sha256": digest(saved),
            "algorithms": algorithms}, standardized


def _reduce(x, scaled, ids, out, progress):
    algorithms = {}
    # Fixed 2D projection supports the required plot; 8D projection quantifies usable compression.
    for name in REDUCERS:
        algorithms[name] = {}
        for dims in (2, 8):
            progress(f"降维：拟合 {name} / {dims}D 并对照聚类")
            start = time.perf_counter()
            if name == "pca":
                fit_input = scaled
                model = PCA(n_components=dims, random_state=42).fit(fit_input)
                coords = model.transform(fit_input)
                reconstructed = model.inverse_transform(coords)
                retention = float(np.sum(model.explained_variance_ratio_))
                retention_name = "explained_variance_ratio"
            else:
                fit_input = x  # NMF requires nonnegative features.
                model = NMF(n_components=dims, init="nndsvda", max_iter=1500, random_state=42).fit(fit_input)
                coords = model.transform(fit_input)
                reconstructed = model.inverse_transform(coords)
                baseline = np.linalg.norm(fit_input - fit_input.mean(axis=0)) ** 2
                retention = float(max(0, 1 - np.linalg.norm(fit_input - reconstructed) ** 2 / baseline)) if baseline else 1.0
                retention_name = "reconstruction_variance_proxy"
            error = float(np.sqrt(np.mean((fit_input - reconstructed) ** 2)))
            model_file = out / f"reduction-{name}-{dims}d.joblib"
            joblib.dump(model, model_file)
            coord_file = out / f"coordinates-{name}-{dims}d.csv"
            pd.DataFrame({"UserID": ids, **{f"component_{i+1}": coords[:, i] for i in range(dims)}}).to_csv(coord_file, index=False)
            cluster_comparison = {}
            for clusterer in CLUSTERERS:
                model2, labels2, params2, seconds2 = _cluster_fit(coords, clusterer, progress)
                assignment = out / f"cluster-{clusterer}-on-{name}-{dims}d.csv"
                pd.DataFrame({"UserID": ids, "cluster": labels2, "noise": labels2 == -1}).to_csv(assignment, index=False)
                cluster_comparison[clusterer] = {"silhouette_reduced": _silhouette(coords, labels2),
                                                 "cluster_count": int(len(set(labels2) - {-1})),
                                                 "noise_count": int(np.sum(labels2 == -1)),
                                                 "parameters": params2, "fit_seconds": round(seconds2, 3),
                                                 "assignment_file": assignment.name}
            algorithms[name][str(dims)] = {"version": f"{VERSION}-{name}-{dims}d", "algorithm": type(model).__name__,
                                          "parameters": model.get_params(), "original_dimensions": int(x.shape[1]),
                                          "dimensions": dims, "retention_metric": retention_name,
                                          "retention": round(retention, 6), "reconstruction_rmse": round(error, 6),
                                          "iterations": int(model.n_iter_) if hasattr(model, "n_iter_") else None,
                                          "converged": bool(model.n_iter_ < model.max_iter) if name == "nmf" else True,
                                          "fit_and_comparison_seconds": round(time.perf_counter() - start, 3),
                                          "model_bytes": model_file.stat().st_size, "coordinates_bytes": coord_file.stat().st_size,
                                          "model_file": model_file.name, "coordinate_file": coord_file.name,
                                          "coordinate_sha256": digest(coord_file), "cluster_comparison": cluster_comparison}
    return {"algorithms": algorithms,
            "comparison_note": "原画像聚类与降维后聚类分别重新拟合；轮廓系数均在训练期用户上计算。PCA 方差比和 NMF 重构方差代理定义不同，不可当作完全等价的保真率。"}


def build_analysis(progress=None, task_id: str | None = None) -> dict:
    progress = progress or (lambda stage: None)
    source, folder = source_report(task_id)
    out = output_dir(source)
    existing = out / "report.json"
    if existing.is_file():
        report = json.loads(existing.read_text(encoding="utf-8"))
        if report.get("status") == "completed" and report.get("data_version") == source["data_version"]:
            try:
                _verify_analysis_files(report, out)
                return report
            except (FileNotFoundError, ValueError):
                pass  # Rebuild incomplete or corrupted derived results from verified source.
    out.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "stage": "verify-source", "source_task_id": source["task_id"],
              "data_version": source["data_version"], "started_at": _now()}
    atomic_json(out / "status.json", status)

    def step(stage):
        status["stage"] = stage
        atomic_json(out / "status.json", status)
        progress(stage)

    try:
        step("load-and-check-chronological-split")
        users, movies, train, val, test = _read_tables(folder, source["time_split"])
        u, m, valid_u, valid_m = _static_arrays(users, movies)
        classification = _classify(train, val, test, u, m, out, step)
        ids, counts, profiles, exposure, liked, _ = _profiles(train, movies, step)
        clustering, scaled = _clusters(profiles, ids, counts, train, movies, exposure, liked, len(users), out, step)
        reduction = _reduce(profiles, scaled, ids, out, step)
        report = {"status": "completed", "version": VERSION, "source_task_id": source["task_id"],
                  "data_version": source["data_version"], "source_rule_version": source["rule_version"],
                  "T1": source["time_split"]["T1"], "T2": source["time_split"]["T2"],
                  "split_counts": source["time_split"]["counts"], "started_at": status["started_at"],
                  "finished_at": _now(), "sklearn_version": sklearn.__version__,
                  "algorithm_selection_basis": "分类：提供的分类课件第24页列出 Logistic Regression 和 Decision Tree，两者比较验证集 AP；聚类：KMeans 与 DBSCAN 在训练期画像上比较轮廓系数及噪声；降维：PCA 与 NMF 比较重构/方差信息和成本。未提供聚类/降维课件，后四种算法按迭代二任务自行选型，不声称已由课件核对。",
                  "classification": classification, "clustering": clustering, "reduction": reduction,
                  "limitations": ["分类预测的是历史评分事件的高分概率，不是未来是否会观看电影。",
                                  "若用户或电影在 T1 前没有评分历史，分类器仍可依据静态元数据预测，但属于冷启动，个体概率更不确定。",
                                  "MovieLens 用户/电影元数据不带时间戳；将其视为训练时可用的静态目录，无法从数据本身验证实际发布时间。",
                                  "聚类/降维为无监督训练期描述；没有独立聚类真值，也不声称测试期群体保持稳定。",
                                  "降维二维图仅为投影；不同降维方法的轴和信息保留指标不可直接同义比较。"]}
        atomic_json(existing, report)
        step("completed")
        status.update(status="completed", finished_at=report["finished_at"])
        atomic_json(out / "status.json", status)
        return report
    except Exception as exc:
        status.update(status="failed", error=str(exc), finished_at=_now())
        atomic_json(out / "status.json", status)
        raise


def _ids(value, multiple=False):
    def checked(item):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError("UserID/MovieID 必须是整数")
        return item

    if multiple:
        if not isinstance(value, list) or not value or len(value) > 100:
            raise ValueError("UserID 集合须为 1–100 个整数")
        return [checked(v) for v in value]
    return checked(value)


def classify(algorithm: str, user_id: int, movie_id: int) -> dict:
    if algorithm not in CLASSIFIERS:
        return {"status": "输入无效", "reason": "未知分类算法"}
    try:
        report, out = latest_analysis()
        uid, mid = _ids(user_id), _ids(movie_id)
        state = joblib.load(out / report["classification"]["feature_file"])
        u, m = state["user_static"], state["movie_static"]
        if uid <= 0 or uid >= len(u) or not np.any(u[uid]) or mid <= 0 or mid >= len(m) or not np.any(m[mid]):
            return {"status": "数据不可用", "reason": "未知 UserID 或 MovieID"}
        features = np.hstack((u[uid], m[mid], _history_features(state["user_history_T1"][uid:uid+1])[0],
                              _history_features(state["movie_history_T1"][mid:mid+1])[0]))[None, :]
        info = report["classification"]["algorithms"][algorithm]
        model = joblib.load(out / info["model_file"])
        if model["scaler"] is not None:
            features = model["scaler"].transform(features)
        probability = float(model["estimator"].predict_proba(features)[0, 1])
        return {"status": "completed", "UserID": uid, "MovieID": mid, "algorithm": algorithm,
                "prediction_high_rating": bool(probability >= 0.5), "probability_rating_at_least_4": round(probability, 6),
                "user_training_ratings": int(state["user_history_T1"][uid, 0]),
                "movie_training_ratings": int(state["movie_history_T1"][mid, 0]),
                "model_version": info["version"], "parameters": info["parameters"],
                "validation": info["validation"], "test": info.get("test", "未评估：非验证集入选模型"),
                "data_version": report["data_version"], "feature_cutoff_T1": report["T1"]}
    except (FileNotFoundError, ValueError) as exc:
        return {"status": "未完成", "reason": str(exc)}


def _custom_cluster(algorithm: str, params: dict, report: dict, out: Path) -> dict:
    if algorithm == "kmeans":
        if set(params) != {"n_clusters"} or not isinstance(params["n_clusters"], int) or not 2 <= params["n_clusters"] <= 20:
            raise ValueError("KMeans 仅接受 n_clusters=2–20")
        config = {"n_clusters": params["n_clusters"], "n_init": 10, "random_state": 42}
    else:
        if not set(params) <= {"eps", "min_samples"} or "eps" not in params:
            raise ValueError("DBSCAN 需指定 eps，可选 min_samples")
        eps = float(params["eps"])
        minimum = int(params.get("min_samples", 10))
        if not 0.1 <= eps <= 30 or not 3 <= minimum <= 100:
            raise ValueError("DBSCAN 需 eps=0.1–30，min_samples=3–100")
        config = {"eps": eps, "min_samples": minimum}
    variant_id = hashlib.sha256(json.dumps({"algorithm": algorithm, "config": config}, sort_keys=True).encode()).hexdigest()[:12]
    variant_dir = out / "variants"
    variant_dir.mkdir(exist_ok=True)
    info_file = variant_dir / f"cluster-{variant_id}.json"
    if info_file.is_file():
        info = json.loads(info_file.read_text(encoding="utf-8"))
        if info.get("data_version") == report["data_version"] and (variant_dir / info["assignment_file"]).is_file():
            return info
    profiles = joblib.load(out / report["clustering"]["profile_file"])
    x = profiles["scaled"]
    start = time.perf_counter()
    model = KMeans(**config).fit(x) if algorithm == "kmeans" else DBSCAN(**config).fit(x)
    labels = model.labels_
    source, folder = source_report(report["source_task_id"])
    _, movies, train, _, _ = _read_tables(folder, source["time_split"])
    ids, counts, _, exposure, liked, _ = _profiles(train, movies, lambda _: None)
    if not np.array_equal(ids, profiles["user_ids"]):
        raise ValueError("数据不可用：画像用户顺序与源数据不一致")
    model_file = variant_dir / f"cluster-{variant_id}.joblib"
    assignment_file = variant_dir / f"cluster-{variant_id}.csv"
    joblib.dump(model, model_file)
    pd.DataFrame({"UserID": ids, "cluster": labels, "noise": labels == -1,
                  "training_ratings": counts}).to_csv(assignment_file, index=False)
    info = {"version": f"{VERSION}-{algorithm}-{variant_id}", "algorithm": type(model).__name__,
            "parameters": config, "silhouette_train": _silhouette(x, labels),
            "cluster_count": int(len(set(labels) - {-1})), "noise_count": int(np.sum(labels == -1)),
            "cluster_sizes": {str(int(label)): int(n) for label, n in Counter(labels).items()},
            "fit_and_summary_seconds": round(time.perf_counter() - start, 3),
            "model_file": model_file.name, "model_sha256": digest(model_file),
            "assignment_file": assignment_file.name, "assignment_sha256": digest(assignment_file),
            "summaries": _cluster_summary(labels, ids, train, movies, exposure, liked),
            "data_version": report["data_version"], "source_task_id": report["source_task_id"]}
    atomic_json(info_file, info)
    return info


def cluster(algorithm: str, user_ids: list[int], parameters: dict | None = None) -> dict:
    if algorithm not in CLUSTERERS:
        return {"status": "输入无效", "reason": "未知聚类算法"}
    try:
        ids = _ids(user_ids, multiple=True)
        report, out = latest_analysis()
        info = (_custom_cluster(algorithm, parameters, report, out) if parameters else
                report["clustering"]["algorithms"][algorithm])
        assignment_dir = out / "variants" if parameters else out
        table = pd.read_csv(assignment_dir / info["assignment_file"]).set_index("UserID")
        user_static = joblib.load(out / report["classification"]["feature_file"])["user_static"]
        rows = []
        for uid in ids:
            if uid not in table.index:
                known = 0 < uid < len(user_static) and np.any(user_static[uid])
                rows.append({"UserID": uid, "status": "数据不可用",
                             "reason": "训练期历史不足 20 条" if known else "未知 UserID"})
                continue
            row = table.loc[uid]
            label = int(row["cluster"])
            rows.append({"UserID": uid, "status": "completed", "cluster": label,
                         "noise": bool(row["noise"]), "training_ratings": int(row["training_ratings"]),
                         "group_summary": info["summaries"].get(str(label)) if label != -1 else None})
        return {"status": "completed", "algorithm": algorithm, "version": info["version"],
                "parameters": info["parameters"], "silhouette_train": info["silhouette_train"],
                "cluster_count": info["cluster_count"], "noise_count": info["noise_count"],
                "cluster_sizes": info.get("cluster_sizes"),
                "users": rows, "data_version": report["data_version"], "cutoff_T1": report["T1"]}
    except (FileNotFoundError, ValueError) as exc:
        return {"status": "未完成", "reason": str(exc)}


def _custom_reduction(algorithm: str, dimensions: int, cluster_algorithm: str,
                      cluster_parameters: dict | None, report: dict, out: Path) -> tuple[dict, Path, dict, Path]:
    if not isinstance(dimensions, int) or not 2 <= dimensions <= 16:
        raise ValueError("目标维数须为 2–16")
    if cluster_parameters:
        base = _custom_cluster(cluster_algorithm, cluster_parameters, report, out)
        base_dir = out / "variants"
        cluster_config = base["parameters"]
    else:
        base = report["clustering"]["algorithms"][cluster_algorithm]
        base_dir = out
        cluster_config = None
    spec = {"algorithm": algorithm, "dimensions": dimensions,
            "cluster_algorithm": cluster_algorithm, "cluster_config": cluster_config}
    variant_id = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    variant_dir = out / "variants"
    variant_dir.mkdir(exist_ok=True)
    meta_file = variant_dir / f"reduction-{variant_id}.json"
    if meta_file.is_file():
        info = json.loads(meta_file.read_text(encoding="utf-8"))
        if (info.get("data_version") == report["data_version"] and
                (variant_dir / info["coordinate_file"]).is_file() and
                (variant_dir / info["assignment_file"]).is_file()):
            return info, variant_dir, base, base_dir
    profiles = joblib.load(out / report["clustering"]["profile_file"])
    matrix = profiles["scaled"] if algorithm == "pca" else profiles["matrix"]
    start = time.perf_counter()
    if algorithm == "pca":
        model = PCA(n_components=dimensions, random_state=42).fit(matrix)
        coordinates = model.transform(matrix)
        retention = float(np.sum(model.explained_variance_ratio_))
        retention_metric = "explained_variance_ratio"
    else:
        model = NMF(n_components=dimensions, init="nndsvda", max_iter=1500, random_state=42).fit(matrix)
        coordinates = model.transform(matrix)
        baseline = np.linalg.norm(matrix - matrix.mean(axis=0)) ** 2
        retention = float(max(0, 1 - np.linalg.norm(matrix - model.inverse_transform(coordinates)) ** 2 / baseline)) if baseline else 1.0
        retention_metric = "reconstruction_variance_proxy"
    error = float(np.sqrt(np.mean((matrix - model.inverse_transform(coordinates)) ** 2)))
    if cluster_config:
        cluster_model = (KMeans(**cluster_config).fit(coordinates) if cluster_algorithm == "kmeans"
                         else DBSCAN(**cluster_config).fit(coordinates))
        reduced_labels = cluster_model.labels_
        reduced_params = cluster_config
    else:
        cluster_model, reduced_labels, reduced_params, _ = _cluster_fit(coordinates, cluster_algorithm, lambda _: None)
    model_file = variant_dir / f"reduction-{variant_id}.joblib"
    coordinate_file = variant_dir / f"coordinates-{variant_id}.csv"
    assignment_file = variant_dir / f"cluster-on-reduction-{variant_id}.csv"
    joblib.dump(model, model_file)
    pd.DataFrame({"UserID": profiles["user_ids"],
                  **{f"component_{i+1}": coordinates[:, i] for i in range(dimensions)}}).to_csv(coordinate_file, index=False)
    pd.DataFrame({"UserID": profiles["user_ids"], "cluster": reduced_labels,
                  "noise": reduced_labels == -1}).to_csv(assignment_file, index=False)
    info = {"version": f"{VERSION}-{algorithm}-{dimensions}d-{variant_id}",
            "algorithm": type(model).__name__, "parameters": model.get_params(),
            "original_dimensions": int(matrix.shape[1]), "dimensions": dimensions,
            "retention_metric": retention_metric, "retention": round(retention, 6),
            "reconstruction_rmse": round(error, 6), "iterations": int(model.n_iter_) if hasattr(model, "n_iter_") else None,
            "converged": bool(model.n_iter_ < model.max_iter) if algorithm == "nmf" else True,
            "fit_and_comparison_seconds": round(time.perf_counter() - start, 3),
            "model_bytes": model_file.stat().st_size, "coordinates_bytes": coordinate_file.stat().st_size,
            "model_file": model_file.name, "model_sha256": digest(model_file),
            "coordinate_file": coordinate_file.name, "coordinate_sha256": digest(coordinate_file),
            "assignment_file": assignment_file.name, "assignment_sha256": digest(assignment_file),
            "cluster_parameters": reduced_params, "silhouette_reduced": _silhouette(coordinates, reduced_labels),
            "cluster_count": int(len(set(reduced_labels) - {-1})), "noise_count": int(np.sum(reduced_labels == -1)),
            "data_version": report["data_version"], "source_task_id": report["source_task_id"]}
    atomic_json(meta_file, info)
    return info, variant_dir, base, base_dir


def reduce(algorithm: str, user_ids: list[int], dimensions: int = 2, cluster_algorithm: str = "kmeans",
           retention_threshold: float | None = None, cluster_parameters: dict | None = None) -> dict:
    if algorithm not in REDUCERS or cluster_algorithm not in CLUSTERERS:
        return {"status": "输入无效", "reason": "未知降维或聚类算法"}
    try:
        ids = _ids(user_ids, multiple=True)
        report, out = latest_analysis()
        choices = report["reduction"]["algorithms"][algorithm]
        if retention_threshold is not None:
            if not 0 < retention_threshold <= 1:
                return {"status": "输入无效", "reason": "信息保留阈值应在 (0,1]"}
            for candidate in (2, 4, 6, 8, 10, 12, 14, 16):
                if str(candidate) in choices and not cluster_parameters:
                    candidate_info = choices[str(candidate)]
                else:
                    candidate_info, _, _, _ = _custom_reduction(
                        algorithm, candidate, cluster_algorithm, cluster_parameters, report, out)
                if candidate_info["retention"] >= retention_threshold:
                    dimensions = candidate
                    break
            else:
                return {"status": "未完成", "reason": "在最多 16 维内未达到所需信息保留阈值"}
        if str(dimensions) in choices and not cluster_parameters:
            info = choices[str(dimensions)]
            coord_dir = out
            comparison = info["cluster_comparison"][cluster_algorithm]
            assignment_dir = out
            assignment_file = comparison["assignment_file"]
            base = report["clustering"]["algorithms"][cluster_algorithm]
            base_dir = out
        else:
            info, coord_dir, base, base_dir = _custom_reduction(
                algorithm, dimensions, cluster_algorithm, cluster_parameters, report, out)
            comparison = {"silhouette_reduced": info["silhouette_reduced"],
                          "cluster_count": info["cluster_count"], "noise_count": info["noise_count"],
                          "parameters": info["cluster_parameters"]}
            assignment_dir = coord_dir
            assignment_file = info["assignment_file"]
        coords = pd.read_csv(coord_dir / info["coordinate_file"]).set_index("UserID")
        assignments = pd.read_csv(assignment_dir / assignment_file).set_index("UserID")
        original = pd.read_csv(base_dir / base["assignment_file"]).set_index("UserID")
        user_static = joblib.load(out / report["classification"]["feature_file"])["user_static"]
        rows = []
        for uid in ids:
            if uid not in coords.index:
                known = 0 < uid < len(user_static) and np.any(user_static[uid])
                rows.append({"UserID": uid, "status": "数据不可用",
                             "reason": "训练期历史不足 20 条" if known else "未知 UserID"})
            else:
                rows.append({"UserID": uid, "status": "completed", "coordinates": [round(float(x), 6) for x in coords.loc[uid]],
                             "cluster_original": int(original.loc[uid, "cluster"]),
                             "cluster_reduced": int(assignments.loc[uid, "cluster"])})
        return {"status": "completed", "algorithm": algorithm, "version": info["version"],
                "dimensions_original": info["original_dimensions"], "dimensions_reduced": dimensions,
                "retention_metric": info["retention_metric"], "retention": info["retention"],
                "reconstruction_rmse": info["reconstruction_rmse"], "parameters": info["parameters"],
                "resource_cost": {"fit_and_comparison_seconds": info["fit_and_comparison_seconds"],
                                  "model_bytes": info["model_bytes"], "coordinates_bytes": info["coordinates_bytes"]},
                "clustering_comparison": {"algorithm": cluster_algorithm, "original_silhouette": base["silhouette_train"],
                                          "reduced_silhouette": comparison["silhouette_reduced"],
                                          "original_parameters": base["parameters"],
                                          "reduced_parameters": comparison["parameters"],
                                          "original_cluster_count": base["cluster_count"],
                                          "reduced_cluster_count": comparison["cluster_count"],
                                          "original_noise_count": base["noise_count"],
                                          "reduced_noise_count": comparison["noise_count"]},
                "users": rows, "coordinate_file": info["coordinate_file"],
                "data_version": report["data_version"], "cutoff_T1": report["T1"]}
    except (FileNotFoundError, ValueError) as exc:
        return {"status": "未完成", "reason": str(exc)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id")
    args = parser.parse_args()
    result = build_analysis(print, args.task_id)
    print(json.dumps({"status": result["status"], "source_task_id": result["source_task_id"],
                      "data_version": result["data_version"]}, ensure_ascii=False))

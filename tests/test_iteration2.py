"""Iteration-2 invariants: real artifacts, version checks and no event leakage."""
import json

import pandas as pd
import pytest

from agent import TOOLS
from iteration2 import (_prior_at_event, classify, cluster, latest_analysis, reduce,
                        source_report)


def test_same_timestamp_events_do_not_see_each_other():
    ratings = pd.DataFrame({"UserID": [1, 1, 1, 2], "MovieID": [10, 11, 12, 10],
                            "Rating": [5, 1, 4, 2], "Timestamp": [100, 100, 101, 101]})
    user_prior = _prior_at_event(ratings, "UserID")
    movie_prior = _prior_at_event(ratings, "MovieID")
    assert user_prior.tolist() == [[0, 0, 0], [0, 0, 0], [2, 6, 1], [0, 0, 0]]
    assert movie_prior.tolist() == [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 5, 1]]


def test_source_task_id_rejects_path_traversal():
    with pytest.raises(ValueError, match="ID"):
        source_report("../../other")


def test_each_algorithm_has_an_independent_agent_tool():
    names = {tool["function"]["name"] for tool in TOOLS}
    assert {"classify_logistic_sgd", "classify_decision_tree", "cluster_kmeans",
            "cluster_dbscan", "reduce_pca", "reduce_nmf"} <= names


def test_registered_artifacts_and_unknown_ids():
    try:
        report, folder = latest_analysis()
    except FileNotFoundError:
        pytest.skip("No completed iteration-2 artifact in this environment")
    assert report["status"] == "completed"
    assert report["data_version"] == source_report()[0]["data_version"]
    assert report["split_counts"]["train"] > 0
    for method in ("logistic_sgd", "decision_tree"):
        predicted = classify(method, 659, 1)
        assert predicted["status"] == "completed"
        assert 0 <= predicted["probability_rating_at_least_4"] <= 1
        assert classify(method, 999999, 1)["status"] == "数据不可用"
    for method in ("kmeans", "dbscan"):
        groups = cluster(method, [659, 1, 999999])
        assert groups["users"][0]["status"] == "completed"
        assert groups["users"][1]["status"] == "数据不可用"
        assert groups["users"][1]["reason"] == "训练期历史不足 20 条"
        assert groups["users"][2]["reason"] == "未知 UserID"
    for method in ("pca", "nmf"):
        coordinates = reduce(method, [659], dimensions=2)
        assert coordinates["status"] == "completed"
        assert len(coordinates["users"][0]["coordinates"]) == 2


def test_report_references_existing_models():
    try:
        report, folder = latest_analysis()
    except FileNotFoundError:
        pytest.skip("No completed iteration-2 artifact in this environment")
    for method in ("logistic_sgd", "decision_tree"):
        assert (folder / report["classification"]["algorithms"][method]["model_file"]).is_file()
    for method in ("pca", "nmf"):
        for dimensions in ("2", "8"):
            item = report["reduction"]["algorithms"][method][dimensions]
            assert (folder / item["model_file"]).is_file()
            assert (folder / item["coordinate_file"]).is_file()


def test_custom_parameters_are_fit_and_versioned():
    try:
        latest_analysis()
    except FileNotFoundError:
        pytest.skip("No completed iteration-2 artifact in this environment")
    custom_group = cluster("kmeans", [659], {"n_clusters": 3})
    assert custom_group["status"] == "completed"
    assert custom_group["cluster_count"] == 3
    assert custom_group["version"].startswith("iteration2-v")
    custom_projection = reduce("pca", [659], dimensions=3, cluster_algorithm="kmeans",
                               cluster_parameters={"n_clusters": 3})
    assert custom_projection["status"] == "completed"
    assert custom_projection["dimensions_reduced"] == 3
    assert custom_projection["clustering_comparison"]["original_cluster_count"] == 3
    assert len(custom_projection["users"][0]["coordinates"]) == 3
    assert cluster("kmeans", [659], {"n_clusters": 100})["status"] == "未完成"

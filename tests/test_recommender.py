from pathlib import Path

import numpy as np
import pandas as pd

from backend.recommender import MovieRecommender


def make_recommender(tmp_path: Path) -> MovieRecommender:
    pd.DataFrame(
        [
            [1, "Toy Story (1995)", "Adventure|Comedy"],
            [2, "Toy Story 2 (1999)", "Animation"],
            [3, "Obscure Film (2001)", "(no genres listed)"],
        ],
        columns=["movieId", "title", "genres"],
    ).to_csv(tmp_path / "movies.csv", index=False)
    pd.DataFrame(
        [[1, 1, 5], [1, 2, 5], [2, 1, 5], [2, 2, 4.5], [3, 3, 5]],
        columns=["userId", "movieId", "rating"],
    ).to_csv(tmp_path / "ratings.csv", index=False)
    return MovieRecommender(tmp_path)


def test_search_is_sorted_and_includes_similarity(tmp_path: Path) -> None:
    recommender = make_recommender(tmp_path)
    results = recommender.search("Toy Story")
    assert results
    assert [item["similarity"] for item in results] == sorted(
        (item["similarity"] for item in results), reverse=True
    )
    assert all("similarity" in item for item in results)


def test_search_handles_case_punctuation_and_partial_titles(tmp_path: Path) -> None:
    recommender = make_recommender(tmp_path)
    assert recommender.search("toy story 1995")[0]["movieId"] == 1
    assert recommender.search("TOY-ST")[0]["movieId"] == 1
    assert all("Trans" not in item["title"] for item in recommender.search("toy story"))


def test_unknown_search_is_empty(tmp_path: Path) -> None:
    assert make_recommender(tmp_path).search("qzxvplmno") == []


def test_obscure_movie_does_not_crash(tmp_path: Path) -> None:
    results = make_recommender(tmp_path).recommend(3)
    assert all(item["movieId"] != 3 for item in results)
    assert all(np.isfinite(item["score"]) for item in results)


def test_recommendation_shape_and_exclusion(tmp_path: Path) -> None:
    results = make_recommender(tmp_path).recommend(1)
    assert results and results[0]["movieId"] != 1
    assert {"movieId", "title", "year", "genres", "score"} <= results[0].keys()


def test_diversity_caps_same_title_family(tmp_path: Path) -> None:
    recommender = make_recommender(tmp_path)
    recommender.movie_info.update(
        {
            4: {"title": "Toy Story 3 (2010)", "genres": "Animation", "year": 2010},
            5: {"title": "The Matrix (1999)", "genres": "Sci-Fi", "year": 1999},
            6: {"title": "Arrival (2016)", "genres": "Sci-Fi", "year": 2016},
        }
    )
    ids = np.array([2, 4, 5, 6])
    scores = np.array([10.0, 9.0, 8.0, 7.0])
    selected = recommender._diverse_order(ids, scores, 1, 4)
    assert list(ids[selected]).count(2) + list(ids[selected]).count(4) == 2
    assert set(ids[selected]) == {2, 4, 5, 6}

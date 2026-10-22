"""Item-based collaborative filtering and title search for MovieLens."""

from __future__ import annotations

from functools import lru_cache
from difflib import SequenceMatcher
import os
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
NON_WORD_RE = re.compile(r"[^a-z0-9 ]")
YEAR_QUERY_RE = re.compile(r"\b(18|19|20)\d{2}\b")
SEQUEL_NUMBER_RE = re.compile(r"\b(\d+)\b")


def clean_title(title: str) -> str:
    """Match the notebook's title normalization exactly."""
    return re.sub(r"[^a-zA-Z0-9 ]", "", str(title))


class MovieRecommender:
    """Memory-efficient search and high-rating collaborative filtering indexes."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        root = Path(data_dir or Path(__file__).resolve().parents[1] / "data")
        self.data_dir = root
        self.movies = pd.read_csv(root / "movies.csv", dtype={"movieId": "int32"})
        self.movies["year"] = self.movies["title"].str.extract(YEAR_RE, expand=False)
        self.movies["year"] = pd.to_numeric(self.movies["year"], errors="coerce").astype("Int16")
        self.movies["clean_title"] = self.movies["title"].map(clean_title)
        self.movies["normalized_title"] = self.movies["title"].map(self._normalize_title)
        self.movie_info: dict[int, dict[str, Any]] = (
            self.movies.set_index("movieId")[["title", "genres", "year"]].to_dict("index")
        )
        self.tmdb_key = os.getenv("TMDB_API_KEY")
        self.tmdb_ids: dict[int, int] = {}
        links_path = root / "links.csv"
        if self.tmdb_key and links_path.exists():
            links = pd.read_csv(links_path, usecols=["movieId", "tmdbId"], dtype={"movieId": "int32", "tmdbId": "Int64"})
            self.tmdb_ids = {
                int(row.movieId): int(row.tmdbId)
                for row in links.itertuples()
                if pd.notna(row.tmdbId)
            }
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.tfidf = self.vectorizer.fit_transform(self.movies["clean_title"])
        self.content_tfidf = self._build_content_matrix(root)

        ratings = pd.read_csv(
            root / "ratings.csv",
            usecols=["userId", "movieId", "rating"],
            dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
        )
        high = ratings.loc[ratings["rating"] > 4, ["userId", "movieId"]]
        self.high_ratings = high
        self.high_rating_counts = high["movieId"].value_counts().to_dict()
        self.movie_to_users = {
            int(movie_id): group["userId"].to_numpy(dtype=np.int32, copy=True)
            for movie_id, group in high.groupby("movieId", sort=False)
        }
        self.user_to_movies = {
            int(user_id): group["movieId"].to_numpy(dtype=np.int32, copy=True)
            for user_id, group in high.groupby("userId", sort=False)
        }
        self.loaded = True

    @lru_cache(maxsize=2048)
    def _poster_url(self, movie_id: int) -> str | None:
        tmdb_id = self.tmdb_ids.get(int(movie_id))
        if not self.tmdb_key or not tmdb_id:
            return None

    @staticmethod
    def _normalize_title(title: str) -> str:
        without_year = YEAR_RE.sub("", str(title))
        return " ".join(NON_WORD_RE.sub(" ", without_year.lower()).split())

    def _build_content_matrix(self, root: Path):
        tags_by_movie: dict[int, str] = {}
        tags_path = root / "tags.csv"
        if tags_path.exists():
            tags = pd.read_csv(tags_path, usecols=["movieId", "tag"], dtype={"movieId": "int32", "tag": "string"})
            tags = tags.dropna(subset=["tag"])
            tags_by_movie = tags.groupby("movieId")["tag"].apply(lambda values: " ".join(values.astype(str))).to_dict()
        documents = [
            f"{str(row.genres).replace('|', ' ')} {tags_by_movie.get(int(row.movieId), '')}"
            for row in self.movies.itertuples()
        ]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.content_vectorizer = vectorizer
        return vectorizer.fit_transform(documents)
        request = Request(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={self.tmdb_key}",
            headers={"User-Agent": "Reelwise/1.0"},
        )
        try:
            with urlopen(request, timeout=2) as response:
                poster_path = json.load(response).get("poster_path")
            return f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
        except (HTTPError, URLError, TimeoutError, ValueError):
            return None

    def _movie_result(self, movie_id: int, score: float | None = None, include_poster: bool = False) -> dict[str, Any]:
        info = self.movie_info.get(int(movie_id), {"title": "Unknown movie", "genres": "(no genres listed)", "year": None})
        genres = info["genres"] or "(no genres listed)"
        result: dict[str, Any] = {
            "movieId": int(movie_id),
            "title": info["title"],
            "year": int(info["year"]) if pd.notna(info["year"]) else None,
            "genres": genres,
        }
        if score is not None:
            result["score"] = round(float(score), 6)
        if include_poster:
            poster = self._poster_url(int(movie_id))
            if poster:
                result["posterUrl"] = poster
        return result

    def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []
        cleaned = clean_title(query)
        if not cleaned.strip():
            return []
        normalized_query = self._normalize_title(query)
        query_year_match = YEAR_QUERY_RE.search(query)
        query_year = int(query_year_match.group()) if query_year_match else None
        query_tokens = set(normalized_query.split())
        tfidf_similarity = cosine_similarity(self.vectorizer.transform([cleaned]), self.tfidf).ravel()
        ranking: list[tuple[float, int]] = []
        for index, row in enumerate(self.movies.itertuples()):
            title = row.normalized_title
            if not title:
                continue
            title_tokens = set(title.split())
            exact = float(title == normalized_query)
            year_bonus = float(query_year is not None and row.year == query_year)
            prefix = float(title.startswith(normalized_query))
            overlap = len(query_tokens & title_tokens) / max(len(query_tokens), 1)
            fuzzy = SequenceMatcher(None, normalized_query, title).ratio()
            token_similarities = [
                max(SequenceMatcher(None, token, candidate).ratio() for candidate in title_tokens)
                for token in query_tokens
            ]
            token_fuzzy = np.mean(token_similarities) if token_similarities else 0.0
            weakest_token = min(token_similarities, default=0.0)
            is_short = len(normalized_query.replace(" ", "")) < 3
            query_has_number = bool(SEQUEL_NUMBER_RE.search(normalized_query))
            sequel_match = (
                SEQUEL_NUMBER_RE.search(title[len(normalized_query):])
                if title.startswith(normalized_query + " ")
                else None
            )
            credible = (
                exact
                or (year_bonus and overlap >= 0.5)
                or (
                    overlap >= 0.5
                    and (
                        prefix
                        or (token_fuzzy >= 0.86 and weakest_token >= 0.78)
                    )
                )
                or (len(query_tokens) == 1 and prefix and len(normalized_query) >= 3)
                or (not is_short and token_fuzzy >= 0.86 and weakest_token >= 0.78 and fuzzy >= 0.60)
            )
            if credible:
                score = (
                    exact * 100
                    + year_bonus * 20
                    + prefix * 15
                    + overlap * 30
                    + token_fuzzy * 20
                    + fuzzy * 5
                    + float(tfidf_similarity[index]) * 5
                )
                if not query_has_number and sequel_match:
                    score += 2 / max(int(sequel_match.group(1)), 1)
                ranking.append((score, index))
        ranking.sort(key=lambda item: (-item[0], int(self.movies.iloc[item[1]]["movieId"])))
        indices = [index for _, index in ranking[:limit]]
        return [
            {
                **self._movie_result(int(self.movies.iloc[index]["movieId"])),
                "similarity": round(float(score), 6),
            }
            for score, index in ranking[:limit]
        ]

    def _family_key(self, movie_id: int, seed_id: int) -> str | None:
        """Group obvious sequels without treating every shared genre as a franchise."""
        seed_title = YEAR_RE.sub("", str(self.movie_info.get(seed_id, {}).get("title", ""))).lower()
        title = YEAR_RE.sub("", str(self.movie_info.get(movie_id, {}).get("title", ""))).lower()
        seed_words = NON_WORD_RE.sub(" ", seed_title).split()
        words = NON_WORD_RE.sub(" ", title).split()
        if not seed_words or not words:
            return None
        if seed_words[0] == "the":
            seed_words = seed_words[1:]
        if words[0] == "the":
            words = words[1:]
        family_size = min(2, len(seed_words))
        family = " ".join(seed_words[:family_size])
        return family if " ".join(words[:family_size]) == family else None

    def _diverse_order(self, candidate_ids: np.ndarray, scores: np.ndarray, seed_id: int, limit: int) -> np.ndarray:
        """Keep at most two obvious same-title sequels in the visible top results."""
        ranked = np.argsort(-scores, kind="stable")
        selected: list[int] = []
        family_counts: dict[str, int] = {}
        deferred: list[int] = []
        for index in ranked:
            family = self._family_key(int(candidate_ids[index]), seed_id)
            if family and family_counts.get(family, 0) >= 2:
                deferred.append(int(index))
                continue
            selected.append(int(index))
            if family:
                family_counts[family] = family_counts.get(family, 0) + 1
            if len(selected) == limit:
                return np.array(selected, dtype=np.intp)
        for index in deferred:
            if len(selected) == limit:
                break
            selected.append(index)
        return np.array(selected, dtype=np.intp)

    def _content_candidates(self, movie_id: int, limit: int = 120) -> tuple[np.ndarray, np.ndarray]:
        seed_index = self.movies.index[self.movies["movieId"] == int(movie_id)]
        if len(seed_index) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float64)
        similarities = cosine_similarity(self.content_tfidf[seed_index[0]], self.content_tfidf).ravel()
        similarities[seed_index[0]] = 0
        available = np.flatnonzero(similarities > 0)
        if not len(available):
            return np.array([], dtype=np.int32), np.array([], dtype=np.float64)
        similarities = similarities[available]
        count = min(limit, len(similarities))
        indices = np.argpartition(similarities, -count)[-count:]
        indices = indices[np.argsort(similarities[indices])[::-1]]
        original_indices = available[indices]
        return self.movies.iloc[original_indices]["movieId"].to_numpy(dtype=np.int32), similarities[indices]

    @lru_cache(maxsize=512)
    def recommend(self, movie_id: int, limit: int = 10) -> list[dict[str, Any]]:
        """Rank high-rating overlaps while widening results beyond one movie franchise."""
        seed_users = self.movie_to_users.get(int(movie_id))
        if seed_users is None or len(seed_users) == 0:
            return []
        candidate_movies = np.concatenate(
            [self.user_to_movies[int(user_id)] for user_id in np.unique(seed_users)]
        )
        candidate_ids, counts = np.unique(candidate_movies, return_counts=True)
        similar_share = counts.astype(np.float64) / len(seed_users)
        # The notebook's fixed 0.1 cutoff is too strict for large audiences:
        # it leaves only sequels for popular franchises. Require at least five
        # similar users where possible, while retaining the original cutoff for
        # small seed audiences.
        min_share = max(0.03, min(0.1, 5 / len(seed_users)))
        eligible = (similar_share >= min_share) & (candidate_ids != int(movie_id))
        candidate_ids, similar_share = candidate_ids[eligible], similar_share[eligible]
        coll_ids = candidate_ids
        coll_scores = np.zeros(len(coll_ids), dtype=np.float64)
        if len(coll_ids):
            all_users: set[int] = set()
            for candidate_id in coll_ids:
                all_users.update(int(user_id) for user_id in self.movie_to_users.get(int(candidate_id), []))
            if all_users:
                all_share = np.array(
                    [len(self.movie_to_users.get(int(candidate_id), [])) / len(all_users) for candidate_id in coll_ids]
                )
                valid = all_share > 0
                coll_scores[valid] = similar_share[valid] / all_share[valid]

        content_ids, content_scores = self._content_candidates(int(movie_id))
        candidate_ids = np.unique(np.concatenate([coll_ids, content_ids])).astype(np.int32)
        if not len(candidate_ids):
            return []
        content_lookup = dict(zip(content_ids.tolist(), content_scores.tolist()))
        coll_lookup = dict(zip(coll_ids.tolist(), coll_scores.tolist()))
        content = np.array([content_lookup.get(int(candidate_id), 0.0) for candidate_id in candidate_ids])
        collaborative = np.array([coll_lookup.get(int(candidate_id), 0.0) for candidate_id in candidate_ids])
        if collaborative.max() > 0:
            collaborative = collaborative / collaborative.max()
        if content.max() > 0:
            content = content / content.max()
        popularity = np.array([np.log1p(self.high_rating_counts.get(int(candidate_id), 0)) for candidate_id in candidate_ids])
        if popularity.max() > 0:
            popularity = popularity / popularity.max()
        if collaborative.max() > 0:
            scores = 0.55 * content + 0.35 * collaborative + 0.10 * popularity
        else:
            scores = 0.85 * content + 0.15 * popularity
        valid = np.isfinite(scores) & (candidate_ids != int(movie_id))
        candidate_ids, scores = candidate_ids[valid], scores[valid]
        if not len(candidate_ids):
            return []
        order = self._diverse_order(candidate_ids, scores, int(movie_id), limit)
        return [self._movie_result(int(candidate_ids[index]), float(scores[index]), include_poster=True) for index in order]


def load_recommender(data_dir: str | Path | None = None) -> MovieRecommender:
    return MovieRecommender(data_dir)

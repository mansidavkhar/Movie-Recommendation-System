"""Flask application serving the movie recommender and static frontend."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

try:
    from backend.recommender import MovieRecommender
except ModuleNotFoundError:  # supports the documented `python backend\app.py` command
    from recommender import MovieRecommender

ROOT = Path(__file__).resolve().parents[1]
app = Flask(__name__, static_folder=str(ROOT / "frontend"), static_url_path="")
recommender: MovieRecommender | None = None
loading = True
load_error: str | None = None


def _load_data() -> None:
    global recommender, loading, load_error
    try:
        recommender = MovieRecommender(ROOT / "data")
    except Exception as exc:  # startup must expose the failure to health/UI
        load_error = str(exc)
    finally:
        loading = False


threading.Thread(target=_load_data, daemon=True).start()


@app.get("/api/health")
def health() -> tuple[Any, int]:
    state = "error" if load_error else ("loading" if loading else "ready")
    return jsonify(
        {
            "status": state,
            "loaded": not loading and recommender is not None,
            "error": load_error,
        }
    ), 503 if load_error else 200


@app.get("/api/search")
def search() -> tuple[Any, int]:
    if load_error:
        return jsonify({"error": load_error, "loading": False}), 503
    if loading:
        # Keep autocomplete polling-friendly: warm-up is an expected state,
        # not a failed search. The frontend retries the latest query.
        return jsonify({"results": [], "loading": True, "message": "Movie data is still warming up."}), 200
    query = request.args.get("q", "")
    return jsonify({"results": recommender.search(query) if recommender else []})


@app.get("/api/recommend/<int:movie_id>")
def recommend(movie_id: int) -> tuple[Any, int]:
    if loading:
        return jsonify({"error": "Movie data is still warming up.", "loading": True}), 503
    if not recommender or movie_id not in recommender.movie_info:
        return jsonify({"error": "That movie was not found."}), 404
    try:
        limit = max(1, min(int(request.args.get("n", 10)), 20))
    except ValueError:
        limit = 10
    results = recommender.recommend(movie_id, limit)
    message = None if results else "This movie does not have enough highly rated viewing overlap yet."
    return jsonify({"movie": recommender._movie_result(movie_id), "recommendations": results, "message": message})


@app.get("/")
def index() -> Any:
    return send_from_directory(ROOT / "frontend", "index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

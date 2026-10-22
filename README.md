# Hybrid Movie Recommendation System

This project is a no-build Flask web application for discovering movies from the MovieLens 25M dataset. It separates two problems that are often confused:

1. **Movie identification:** accurately find the movie the user typed.
2. **Movie recommendation:** find related films using metadata and audience behavior, even when their titles share no words.

## Motivation and features

The original notebook searched titles with TF-IDF and recommended movies using high-rating user overlap. That produced weak search behavior for partial or misspelled queries and could overproduce sequels. Reelwise now provides:

- Exact, year-aware, prefix, token, conservative fuzzy, and TF-IDF fallback title search
- Relevance gates that reject weak matches instead of returning arbitrary titles
- Genre and MovieLens tag content similarity
- High-rating collaborative filtering with adaptive support
- Normalized hybrid ranking and popularity adjustment
- Diversity-aware reranking with a same-family cap
- Responsive autocomplete, keyboard navigation, loading states, history, and empty states
- Optional TMDB posters when `TMDB_API_KEY` is configured

## Architecture

```text
Browser input
  -> debounced /api/search?q=...
  -> MovieRecommender.search()
  -> ranked movie candidates
  -> user selects movieId
  -> /api/recommend/<movie_id>
  -> content + collaborative + popularity hybrid ranker
  -> frontend recommendation cards
```

The active implementation is [backend/app.py](backend/app.py), which imports [backend/recommender.py](backend/recommender.py). The frontend is served directly from [frontend/](frontend/), and the original notebook is retained in [notebooks/](notebooks/).

## Dataset

The local MovieLens files are stored in [data/](data/):

You can download the dataset here: https://grouplens.org/datasets/movielens/25m/

- `movies.csv`: `movieId,title,genres`
- `ratings.csv`: `userId,movieId,rating,timestamp`
- `tags.csv`: `userId,movieId,tag,timestamp`
- `links.csv`: `movieId,imdbId,tmdbId`
- `genome-scores.csv` and `genome-tags.csv`: retained dataset files not required by the current runtime

The runtime reads only the movie metadata, tags, and `userId,movieId,rating` columns needed for the indexes. The large CSV files are ignored by Git through [.gitignore](.gitignore).

## Installation and launch

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python backend\app.py
```

Open <http://127.0.0.1:5000>. The first startup builds the title/content matrices and high-rating indexes. The interface shows a warm-up state until `/api/health` reports `status: "ready"`.

Optional poster configuration:

```powershell
$env:TMDB_API_KEY = "your-key"
python backend\app.py
```

The key is never stored in source. Without it, deterministic gradient poster cards are used.

## API

### `GET /api/health`

Returns:

```json
{"status":"ready","loaded":true,"error":null}
```

`status` is `loading`, `ready`, or `error`. During normal startup, `/api/search` returns HTTP 200 with `loading: true`; the frontend retries the latest query. Startup failures return HTTP 503.

### `GET /api/search?q=toy%20story`

Returns up to eight ordered candidates:

```json
{
  "results": [
    {
      "movieId": 1,
      "title": "Toy Story (1995)",
      "year": 1995,
      "genres": "Adventure|Animation|Children|Comedy|Fantasy",
      "similarity": 190.2
    }
  ]
}
```

### `GET /api/recommend/1?n=10`

Returns the selected movie and recommendations:

```json
{
  "movie": {"movieId": 1, "title": "Toy Story (1995)", "year": 1995, "genres": "..."},
  "recommendations": [
    {"movieId": 3114, "title": "Toy Story 2 (1999)", "year": 1999, "genres": "...", "score": 0.82}
  ],
  "message": null
}
```

## Search algorithm

Movie search does not use recommendation similarity. Titles are normalized by lowercasing, removing the trailing year for comparison, replacing punctuation with spaces, and collapsing whitespace. Candidate scores prioritize:

1. Exact normalized title
2. Matching release year
3. Prefix and full token overlap
4. Token-aware fuzzy similarity for spelling mistakes such as `toy stroy`
5. Whole-title fuzzy and title TF-IDF similarity

Candidates must pass a credibility gate. An arbitrary title with only a small character overlap is excluded. Empty, unknown, and non-Latin-only weak matches do not become strong English results. Final results are explicitly sorted by score with movie ID as a deterministic tie-breaker. Sequel numbers receive a small ordering preference when the query does not specify a number, so `Toy Story (1995)`, `Toy Story 2`, `Toy Story 3`, and `Toy Story 4` appear naturally.

## Recommendation methodology

### Content component

Each movie profile combines its pipe-separated genres and all available MovieLens tags. A word/bigram TF-IDF matrix is built once at startup. Content similarity is cosine similarity between the selected movie vector and candidate vectors.

### Collaborative component

Ratings are loaded as `int32`/`float32`, filtered once to `rating > 4`, and indexed by movie-to-users and user-to-movies. Candidate support is compared with the broader high-rating audience, preserving the notebook’s popularity penalty while allowing an adaptive minimum share for large audiences. Missing collaborative evidence produces a safe content fallback.

### Hybrid and diversity ranking

After independent normalization to `[0,1]`, the configured score is:

```text
0.55 * content + 0.35 * collaborative + 0.10 * popularity
```

When no collaborative evidence exists:

```text
0.85 * content + 0.15 * popularity
```

The selected movie is always excluded. Results are finite, descending, cached per movie ID, and lightly reranked so no obvious same-title family consumes the whole list.

## Frontend behavior

The vanilla JavaScript frontend debounces typing, cancels superseded fetch requests with `AbortController`, ignores stale responses, preserves backend order, and passes the selected `movieId` to the recommendation endpoint. It displays title, year, genres, match bars, skeleton loaders, warm-up status, error/empty states, and an exploration history.

## Testing

```powershell
pytest -q
```

The tests cover:

- Sorted search output and similarity fields
- Case, punctuation, partial title, and misspelling behavior
- Unknown queries
- Sparse/obscure movies
- Finite scores and self-exclusion
- Same-title family diversity

For a real-data smoke test:

```powershell
python backend\app.py
Invoke-WebRequest http://127.0.0.1:5000/api/health
Invoke-WebRequest "http://127.0.0.1:5000/api/search?q=transformers"
Invoke-WebRequest "http://127.0.0.1:5000/api/recommend/53996?n=10"
```

## Example behavior

- `toy story` prioritizes `Toy Story (1995)`, then numbered sequels and related title variants.
- `toy stroy` uses conservative token-aware fuzzy matching to recover Toy Story.
- `transformers` identifies the Transformers titles, while recommendations can include cross-title action/science-fiction films such as Pacific Rim when supported by metadata and ratings.
- An unknown query returns an empty result list rather than unrelated films.
- A movie with no useful rating overlap receives content-based recommendations where metadata supports them.

## Limitations and future work

MovieLens tags are user-supplied and unevenly distributed; there are no plot summaries, cast, or director fields in the runtime inputs. Content recommendations are therefore genre/tag based, not semantic plot recommendations. The full in-memory startup process is appropriate for this local project but would need a persisted index and production WSGI server for deployment. Future improvements could add offline ranking evaluation, richer licensed metadata, semantic embeddings, user profiles, feedback collection, and a persisted vector/index store.

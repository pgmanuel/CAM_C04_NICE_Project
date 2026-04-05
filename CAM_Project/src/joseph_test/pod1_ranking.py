"""
pod1_ranking.py
---------------
Pod 1 Ranking Model — placeholder file ready for your team's code.

HOW THIS CONNECTS TO app_v2.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
In app_v2.py, Section A contains the function _rank_pod1().
Right now that function is a placeholder that does nothing clever.

To connect this file:

  STEP 1 — At the very top of app_v2.py, add this import:
      from pod1_ranking import rank_candidates as _pod1_ranker

  STEP 2 — Inside _rank_pod1() in Section A, replace the
  placeholder body with a single line:
      return _pod1_ranker(candidates, query)

That is the only change needed in app_v2.py.

INPUT CONTRACT (what app_v2.py sends to this file)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  candidates: list[dict]
      A list of clinical code dictionaries. Each dict always contains:
          "MedCodeId"  str    The SNOMED/clinical code identifier
          "term"       str    The human-readable code description
          "score"      float  Initial match score from CSV (0.0 to 1.0)

  query: str
      The plain-English query the analyst typed into the chatbot.
      Example: "obesity with type 2 diabetes"

OUTPUT CONTRACT (what this file must return to app_v2.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  list[dict]
      The same list of dicts, but each dict must now also contain:
          "rank"              int    Position in ranked order (1 = most relevant)
          "confidence_score"  float  Your model's confidence 0.0 to 1.0
          "ranked_by"         str    A label shown in the UI, e.g. "pod1_tfidf"

      You may also add additional fields — they will be passed through
      to the UI output unchanged, which can be useful for debugging.
      Do not remove or rename the original fields.

EXAMPLE of what a correctly shaped output looks like:
    Input candidates:
        [
            {"MedCodeId": "44054006", "term": "Type 2 diabetes mellitus", "score": 0.8},
            {"MedCodeId": "38341003", "term": "Hypertension", "score": 0.7},
        ]

    Expected output:
        [
            {"MedCodeId": "44054006", "term": "Type 2 diabetes mellitus",
             "score": 0.8, "rank": 1, "confidence_score": 0.91, "ranked_by": "pod1_tfidf"},
            {"MedCodeId": "38341003", "term": "Hypertension",
             "score": 0.7, "rank": 2, "confidence_score": 0.74, "ranked_by": "pod1_tfidf"},
        ]
"""

from __future__ import annotations


def rank_candidates(candidates: list[dict], query: str) -> list[dict]:
    """
    Pod 1 ranking function.

    Replace this entire function body with your ranking logic.
    The function signature (name, parameters, return type) must not change.

    Parameters
    ----------
    candidates : list[dict]
        Clinical code candidates from the CSV search.
        Each dict has: MedCodeId (str), term (str), score (float).

    query : str
        The user's plain-English clinical query.

    Returns
    -------
    list[dict]
        Same candidates with rank, confidence_score, and ranked_by added.
    """

    # ─── PLACEHOLDER — replace with your ranking logic below ──────────────
    #
    # Ideas for what your ranking could do here:
    #   - TF-IDF similarity between query and term descriptions
    #   - BM25 or cosine similarity using sentence embeddings
    #   - A trained sklearn model (e.g. RandomForestClassifier)
    #   - Rules based on QOF indicator membership
    #   - Any combination of the above
    #
    # Your function can import anything it needs at the top of this file.
    # Example:
    #   from sklearn.feature_extraction.text import TfidfVectorizer
    #   from sklearn.metrics.pairwise import cosine_similarity
    # ──────────────────────────────────────────────────────────────────────

    ranked = []
    for i, candidate in enumerate(candidates):
        ranked.append({
            **candidate,                      # preserve all original fields
            "rank":             i + 1,        # replace with your computed rank
            "confidence_score": round(candidate.get("score", 0.5), 3),  # replace with your score
            "ranked_by":        "pod1_placeholder",  # replace with your model name
        })

    return ranked

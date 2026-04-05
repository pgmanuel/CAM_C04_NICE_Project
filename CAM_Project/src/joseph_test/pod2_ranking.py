"""
pod2_ranking.py
---------------
Pod 2 Ranking Model — placeholder file ready for your team's code.

HOW THIS CONNECTS TO app_v2.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
In app_v2.py, Section A contains the function _rank_pod2().
Right now that function is a placeholder.

To connect this file:

  STEP 1 — At the very top of app_v2.py, add this import:
      from pod2_ranking import rank_candidates as _pod2_ranker

  STEP 2 — Inside _rank_pod2() in Section A, replace the
  placeholder body with a single line:
      return _pod2_ranker(candidates, query)

That is the only change needed in app_v2.py.

INPUT / OUTPUT CONTRACT
━━━━━━━━━━━━━━━━━━━━━━━
Identical to pod1_ranking.py — see that file for full documentation.
The contract is the same for all ranking modules so that any module
can be plugged into any slot without changing app_v2.py.
"""

from __future__ import annotations


def rank_candidates(candidates: list[dict], query: str) -> list[dict]:
    """
    Pod 2 ranking function.

    Replace this entire function body with your ranking logic.
    The function signature must not change.

    Parameters
    ----------
    candidates : list[dict]
        Each dict has: MedCodeId (str), term (str), score (float).

    query : str
        The user's plain-English clinical query.

    Returns
    -------
    list[dict]
        Same candidates with rank, confidence_score, and ranked_by added.
    """

    # ─── PLACEHOLDER — replace with your ranking logic below ──────────────
    ranked = []
    for i, candidate in enumerate(candidates):
        ranked.append({
            **candidate,
            "rank":             i + 1,
            "confidence_score": round(candidate.get("score", 0.5), 3),
            "ranked_by":        "pod2_placeholder",
        })

    return ranked

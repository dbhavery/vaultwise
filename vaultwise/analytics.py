"""Analytics module: knowledge gaps, usage stats, overview metrics."""

import json
from collections import defaultdict
from datetime import datetime, timezone

from vaultwise.database import get_connection


def get_overview() -> dict:
    """Get overview statistics for the dashboard.

    Returns:
        Dict with total_docs, total_questions, avg_confidence, gaps_count, questions_today.
    """
    conn = get_connection()
    try:
        total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        total_questions = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]

        avg_row = conn.execute("SELECT AVG(confidence) FROM questions").fetchone()
        avg_confidence = round(avg_row[0] or 0.0, 3)

        gaps_count = conn.execute(
            "SELECT COUNT(*) FROM knowledge_gaps WHERE status = 'open'"
        ).fetchone()[0]

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        questions_today = conn.execute(
            "SELECT COUNT(*) FROM questions WHERE created_at LIKE ?",
            (f"{today}%",),
        ).fetchone()[0]

        return {
            "total_docs": total_docs,
            "total_questions": total_questions,
            "avg_confidence": avg_confidence,
            "gaps_count": gaps_count,
            "questions_today": questions_today,
        }
    finally:
        conn.close()


def get_knowledge_gaps(limit: int = 50, offset: int = 0) -> list[dict]:
    """Get knowledge gaps sorted by frequency (highest first).

    Args:
        limit: Max results.
        offset: Pagination offset.

    Returns:
        List of knowledge gap dicts.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM knowledge_gaps ORDER BY frequency DESC, last_asked DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_gap_status(gap_id: str, status: str) -> bool:
    """Update the status of a knowledge gap.

    Args:
        gap_id: The knowledge gap ID.
        status: New status (open, addressed, dismissed).

    Returns:
        True if the gap existed and was updated.
    """
    if status not in ("open", "addressed", "dismissed"):
        raise ValueError(f"Invalid gap status: {status}")

    conn = get_connection()
    try:
        row = conn.execute("SELECT id FROM knowledge_gaps WHERE id = ?", (gap_id,)).fetchone()
        if row is None:
            return False
        conn.execute(
            "UPDATE knowledge_gaps SET status = ? WHERE id = ?",
            (status, gap_id),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def get_usage_stats(days: int = 7) -> dict:
    """Get usage statistics for the specified number of days.

    Args:
        days: Number of days to look back.

    Returns:
        Dict with queries_per_day (list of {date, count}) and total_actions.
    """
    conn = get_connection()
    try:
        # Get all usage log entries
        rows = conn.execute(
            "SELECT action, created_at FROM usage_log ORDER BY created_at DESC"
        ).fetchall()

        total_actions = len(rows)

        # Aggregate by date
        daily_counts: defaultdict[str, int] = defaultdict(int)
        for row in rows:
            date_str = row["created_at"][:10]  # YYYY-MM-DD
            daily_counts[date_str] += 1

        # Get last N days, filling gaps with zeros
        today = datetime.now(timezone.utc)
        queries_per_day: list[dict] = []
        for i in range(days - 1, -1, -1):
            from datetime import timedelta
            day = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            queries_per_day.append({
                "date": day,
                "count": daily_counts.get(day, 0),
            })

        return {
            "queries_per_day": queries_per_day,
            "total_actions": total_actions,
        }
    finally:
        conn.close()

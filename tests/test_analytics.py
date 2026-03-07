"""Tests for the analytics module."""

import json
from uuid import uuid4
from datetime import datetime, timezone

from vaultwise.database import get_connection, init_db


class TestOverview:
    """Tests for the overview analytics endpoint."""

    def test_overview_stats_correct(self, seeded_client):
        """Overview should return correct aggregate stats."""
        resp = seeded_client.get("/api/analytics/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_docs" in data
        assert "total_questions" in data
        assert "avg_confidence" in data
        assert "gaps_count" in data
        assert "questions_today" in data
        assert data["total_docs"] >= 5  # seed has 5 docs
        assert data["total_questions"] >= 10  # seed has 10 questions

    def test_overview_empty_db(self, client):
        """Overview on empty DB should return zeros."""
        resp = client.get("/api/analytics/overview")
        data = resp.json()
        assert data["total_docs"] == 0
        assert data["total_questions"] == 0
        assert data["avg_confidence"] == 0.0


class TestKnowledgeGaps:
    """Tests for knowledge gap tracking."""

    def test_gaps_tracked_and_ranked(self, seeded_client):
        """Gaps should be sorted by frequency descending."""
        resp = seeded_client.get("/api/analytics/gaps")
        assert resp.status_code == 200
        gaps = resp.json()
        assert len(gaps) >= 5  # seed has 5 gaps
        # Check sorted by frequency
        freqs = [g["frequency"] for g in gaps]
        assert freqs == sorted(freqs, reverse=True)

    def test_gap_has_expected_fields(self, seeded_client):
        """Each gap should have all required fields."""
        gaps = seeded_client.get("/api/analytics/gaps").json()
        for g in gaps:
            assert "id" in g
            assert "topic" in g
            assert "frequency" in g
            assert "status" in g
            assert "last_asked" in g

    def test_update_gap_status(self, seeded_client):
        """Updating a gap status should persist."""
        gaps = seeded_client.get("/api/analytics/gaps").json()
        gap_id = gaps[0]["id"]
        resp = seeded_client.patch(f"/api/analytics/gaps/{gap_id}?status=addressed")
        assert resp.status_code == 200
        # Verify update
        updated_gaps = seeded_client.get("/api/analytics/gaps").json()
        gap = next(g for g in updated_gaps if g["id"] == gap_id)
        assert gap["status"] == "addressed"

    def test_update_gap_invalid_status(self, seeded_client):
        """Updating with invalid status should fail."""
        gaps = seeded_client.get("/api/analytics/gaps").json()
        gap_id = gaps[0]["id"]
        resp = seeded_client.patch(f"/api/analytics/gaps/{gap_id}?status=invalid")
        assert resp.status_code == 400


class TestUsageStats:
    """Tests for usage logging and stats."""

    def test_usage_logged(self, seeded_client):
        """Usage stats should reflect logged activity."""
        resp = seeded_client.get("/api/analytics/usage?days=7")
        assert resp.status_code == 200
        data = resp.json()
        assert "queries_per_day" in data
        assert "total_actions" in data
        assert len(data["queries_per_day"]) == 7
        assert data["total_actions"] >= 20  # seed has 20 usage entries

    def test_usage_has_dates(self, seeded_client):
        """Each usage entry should have date and count."""
        data = seeded_client.get("/api/analytics/usage?days=7").json()
        for entry in data["queries_per_day"]:
            assert "date" in entry
            assert "count" in entry
            assert isinstance(entry["count"], int)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check(self, client):
        """Health endpoint should return status and counts."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert "documents" in data
        assert "chunks" in data

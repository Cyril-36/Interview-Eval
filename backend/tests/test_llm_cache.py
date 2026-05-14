"""Tests for the SQLite-backed LLM response cache."""
import time
from unittest.mock import AsyncMock, patch

import pytest

from app.services.llm_cache import LLMCache, make_key


@pytest.fixture()
def cache(tmp_path):
    db = tmp_path / "llm_cache.db"
    c = LLMCache(db_path=db, default_ttl_seconds=60)
    yield c
    c.close()


class TestMakeKey:
    def test_stable_for_equivalent_inputs(self):
        assert make_key("Q", "A", "B") == make_key("q", "a", "b")
        assert make_key("hello world") == make_key("  HELLO   WORLD  ")

    def test_differs_when_parts_differ(self):
        assert make_key("q", "a") != make_key("q", "b")

    def test_part_boundaries_are_respected(self):
        # Joining should not collide with concatenation.
        assert make_key("ab", "c") != make_key("a", "bc")


class TestLLMCache:
    def test_miss_returns_none(self, cache):
        assert cache.get("ns", "missing") is None
        assert cache.stats.misses == 1
        assert cache.stats.hits == 0

    def test_set_then_get_round_trips_value(self, cache):
        payload = {"score": 0.8, "reason": "ok"}
        cache.set("ns", "k1", payload)
        result = cache.get("ns", "k1")
        assert result == payload
        assert cache.stats.hits == 1
        assert cache.stats.sets == 1

    def test_namespace_isolation(self, cache):
        cache.set("ns_a", "k", {"v": 1})
        assert cache.get("ns_b", "k") is None
        assert cache.get("ns_a", "k") == {"v": 1}

    def test_ttl_expires_entries(self, cache):
        cache.set("ns", "k", {"v": 1}, ttl_seconds=0)
        # Even with ttl=0 it should be expired (expires_at == now).
        time.sleep(0.01)
        assert cache.get("ns", "k") is None

    def test_purge_expired_removes_entries(self, cache):
        cache.set("ns", "fresh", {"v": 1}, ttl_seconds=60)
        cache.set("ns", "stale", {"v": 2}, ttl_seconds=0)
        time.sleep(0.01)
        removed = cache.purge_expired()
        assert removed == 1
        assert cache.get("ns", "fresh") == {"v": 1}

    def test_persists_across_instances(self, tmp_path):
        db = tmp_path / "shared.db"
        c1 = LLMCache(db_path=db)
        c1.set("ns", "k", {"v": "hello"})
        c1.close()

        c2 = LLMCache(db_path=db)
        try:
            assert c2.get("ns", "k") == {"v": "hello"}
        finally:
            c2.close()


class TestLLMScorerCaching:
    """End-to-end: llm_scorer.score should hit cache on the second call."""

    @pytest.mark.asyncio
    async def test_second_call_is_cache_hit(self, tmp_path):
        from app.services import llm_cache
        from app.services.scoring import llm_scorer

        llm_cache.reset_cache_for_tests(tmp_path / "llm_cache.db")

        fake_response = type("R", (), {
            "choices": [type("C", (), {
                "message": type("M", (), {
                    "content": (
                        '{"correctness": 80, "completeness": 70,'
                        ' "clarity": 60, "depth": 50,'
                        ' "reason": "decent"}'
                    )
                })()
            })()],
        })()

        call_count = {"n": 0}

        async def fake_run_in_executor(_executor, fn, *args, **kwargs):
            call_count["n"] += 1
            return fake_response

        # Patch the event loop's run_in_executor where llm_scorer calls it.
        with patch.object(
            llm_scorer, "get_settings",
            return_value=type("S", (), {
                "GROQ_API_KEY": "fake-key",
                "LLM_CACHE_ENABLED": True,
                "LLM_CACHE_TTL_SECONDS": 60,
            })(),
        ), patch(
            "asyncio.get_event_loop",
        ) as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                side_effect=fake_run_in_executor,
            )

            first = await llm_scorer.score(
                candidate_answer="Trees use recursion.",
                ideal_answer="Trees rely on recursive traversal.",
                question="How do you traverse a tree?",
            )
            second = await llm_scorer.score(
                candidate_answer="Trees use recursion.",
                ideal_answer="Trees rely on recursive traversal.",
                question="How do you traverse a tree?",
            )

        assert first.normalized_score == pytest.approx(second.normalized_score)
        assert first.correctness == pytest.approx(second.correctness)
        assert call_count["n"] == 1, "LLM should only be invoked once"

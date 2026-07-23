"""
tests/test_phase10d2_runtime_lifecycle.py

Phase 10D2 lifecycle tests: runtime-path resolution, idempotent knowledge
bootstrap, and scheduler lifecycle. No real Chroma/Redis/MySQL/Scheduler.
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Runtime paths ──────────────────────────────────────────────────────────────


class TestRuntimePaths:
    def test_default_chroma_dir_is_absolute(self, monkeypatch):
        """Default chroma_dir() is an absolute path independent of cwd."""
        monkeypatch.delenv("CHROMA_PERSIST_DIR", raising=False)
        from core.runtime_paths import chroma_dir
        p = chroma_dir()
        assert p.is_absolute(), f"expected absolute path, got {p}"

    def test_env_var_overrides_chroma_dir(self, monkeypatch, tmp_path):
        """CHROMA_PERSIST_DIR env var is returned by chroma_dir()."""
        custom = str(tmp_path / "my_chroma")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", custom)
        from core.runtime_paths import chroma_dir
        assert chroma_dir() == Path(custom)

    def test_ensure_runtime_dirs_creates_dirs(self, monkeypatch, tmp_path):
        """ensure_runtime_dirs() creates chroma, sqlite parent, dead-letter parent."""
        chroma = tmp_path / "vol" / "chroma"
        sqlite = tmp_path / "vol" / "users.db"
        dl = tmp_path / "vol" / "dead_letter.db"
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(chroma))
        monkeypatch.setenv("SQLITE_DB_PATH", str(sqlite))
        monkeypatch.setenv("DEAD_LETTER_DB_PATH", str(dl))

        from core.runtime_paths import ensure_runtime_dirs
        ensure_runtime_dirs()

        assert chroma.is_dir()
        assert sqlite.parent.is_dir()
        assert dl.parent.is_dir()

    def test_static_seed_paths_not_cwd_dependent(self, monkeypatch):
        """Default nutrition_dir() is absolute (not relative to cwd)."""
        monkeypatch.delenv("NUTRITION_DIR", raising=False)
        from core.runtime_paths import nutrition_dir
        p = nutrition_dir()
        assert p.is_absolute(), f"expected absolute path, got {p}"

    def test_volume_and_seed_paths_can_be_separate(self, monkeypatch, tmp_path):
        """Runtime paths and seed paths can point to independent directories."""
        runtime = tmp_path / "runtime-data"
        seeds = tmp_path / "seeds"
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(runtime / "chroma"))
        monkeypatch.setenv("NUTRITION_DIR", str(seeds / "nutrition"))

        from core.runtime_paths import chroma_dir, nutrition_dir
        assert chroma_dir() != nutrition_dir()
        assert "runtime-data" in str(chroma_dir())
        assert "seeds" in str(nutrition_dir())


# ── Nutrition bootstrap ────────────────────────────────────────────────────────


class TestNutritionBootstrap:
    def _make_doc_dir(self, tmp_path: Path) -> str:
        d = tmp_path / "nutrition"
        d.mkdir()
        (d / "test_diet.txt").write_text("营养测试内容\n" * 20, encoding="utf-8")
        return str(d)

    def test_empty_collection_loads_docs(self, monkeypatch, tmp_path):
        """replace=False with empty collection: documents are added."""
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        monkeypatch.setattr("tools.nutrition._nutrition_collection", mock_col)

        from tools.nutrition import load_nutrition_documents
        result = load_nutrition_documents(self._make_doc_dir(tmp_path), replace=False)

        assert result > 0
        mock_col.add.assert_called_once()

    def test_nonempty_replace_false_no_delete(self, monkeypatch, tmp_path):
        """replace=False with non-empty collection: delete is NOT called."""
        mock_col = MagicMock()
        mock_col.count.return_value = 5
        monkeypatch.setattr("tools.nutrition._nutrition_collection", mock_col)

        from tools.nutrition import load_nutrition_documents
        result = load_nutrition_documents(self._make_doc_dir(tmp_path), replace=False)

        assert result == 0
        mock_col.delete.assert_not_called()

    def test_nonempty_replace_false_no_add(self, monkeypatch, tmp_path):
        """replace=False with non-empty collection: add is NOT called."""
        mock_col = MagicMock()
        mock_col.count.return_value = 5
        monkeypatch.setattr("tools.nutrition._nutrition_collection", mock_col)

        from tools.nutrition import load_nutrition_documents
        load_nutrition_documents(self._make_doc_dir(tmp_path), replace=False)

        mock_col.add.assert_not_called()

    def test_replace_true_rebuilds(self, monkeypatch, tmp_path):
        """replace=True with non-empty collection: existing ids are deleted, new docs added."""
        mock_col = MagicMock()
        mock_col.count.return_value = 3
        mock_col.get.return_value = {"ids": ["old_0", "old_1", "old_2"]}
        monkeypatch.setattr("tools.nutrition._nutrition_collection", mock_col)

        from tools.nutrition import load_nutrition_documents
        result = load_nutrition_documents(self._make_doc_dir(tmp_path), replace=True)

        mock_col.delete.assert_called_once_with(ids=["old_0", "old_1", "old_2"])
        mock_col.add.assert_called_once()
        assert result > 0

    def test_missing_dir_no_delete(self, monkeypatch, tmp_path):
        """replace=True with missing doc_dir: delete is NOT called."""
        mock_col = MagicMock()
        mock_col.count.return_value = 3
        mock_col.get.return_value = {"ids": ["id_0"]}
        monkeypatch.setattr("tools.nutrition._nutrition_collection", mock_col)

        from tools.nutrition import load_nutrition_documents
        result = load_nutrition_documents(str(tmp_path / "nonexistent"), replace=True)

        assert result == 0
        mock_col.delete.assert_not_called()

    def test_empty_dir_no_delete(self, monkeypatch, tmp_path):
        """replace=True with empty doc_dir (no .txt): delete is NOT called."""
        empty_dir = tmp_path / "empty_nutrition"
        empty_dir.mkdir()
        mock_col = MagicMock()
        mock_col.count.return_value = 3
        mock_col.get.return_value = {"ids": ["id_0"]}
        monkeypatch.setattr("tools.nutrition._nutrition_collection", mock_col)

        from tools.nutrition import load_nutrition_documents
        result = load_nutrition_documents(str(empty_dir), replace=True)

        assert result == 0
        mock_col.delete.assert_not_called()


# ── Food Safety bootstrap ──────────────────────────────────────────────────────


class TestFoodSafetyBootstrap:
    def _make_doc_dir(self, tmp_path: Path) -> str:
        d = tmp_path / "food_safety"
        d.mkdir()
        (d / "safety_tips.txt").write_text("食品安全测试内容\n" * 20, encoding="utf-8")
        return str(d)

    def test_empty_collection_loads_docs(self, monkeypatch, tmp_path):
        """replace=False with empty collection: documents are added."""
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        monkeypatch.setattr("tools.nutrition._food_safety_collection", mock_col)

        from tools.nutrition import load_food_safety_documents
        result = load_food_safety_documents(self._make_doc_dir(tmp_path), replace=False)

        assert result > 0
        mock_col.add.assert_called_once()

    def test_nonempty_skips(self, monkeypatch, tmp_path):
        """replace=False with non-empty collection: returns 0, no mutation."""
        mock_col = MagicMock()
        mock_col.count.return_value = 7
        monkeypatch.setattr("tools.nutrition._food_safety_collection", mock_col)

        from tools.nutrition import load_food_safety_documents
        result = load_food_safety_documents(self._make_doc_dir(tmp_path), replace=False)

        assert result == 0
        mock_col.delete.assert_not_called()
        mock_col.add.assert_not_called()

    def test_replace_true_rebuilds(self, monkeypatch, tmp_path):
        """replace=True: existing ids are deleted, new docs added."""
        mock_col = MagicMock()
        mock_col.count.return_value = 2
        mock_col.get.return_value = {"ids": ["a_0", "a_1"]}
        monkeypatch.setattr("tools.nutrition._food_safety_collection", mock_col)

        from tools.nutrition import load_food_safety_documents
        result = load_food_safety_documents(self._make_doc_dir(tmp_path), replace=True)

        mock_col.delete.assert_called_once_with(ids=["a_0", "a_1"])
        mock_col.add.assert_called_once()
        assert result > 0

    def test_missing_dir_safe(self, monkeypatch, tmp_path):
        """replace=True with missing dir: no delete, returns 0."""
        mock_col = MagicMock()
        mock_col.count.return_value = 2
        mock_col.get.return_value = {"ids": ["a_0"]}
        monkeypatch.setattr("tools.nutrition._food_safety_collection", mock_col)

        from tools.nutrition import load_food_safety_documents
        result = load_food_safety_documents(str(tmp_path / "no_such_dir"), replace=True)

        assert result == 0
        mock_col.delete.assert_not_called()


# ── Recipe ────────────────────────────────────────────────────────────────────


class TestRecipeBootstrap:
    def test_empty_collection_seeds(self, tmp_path):
        """init_chroma with empty collection triggers _seed_recipes (add is called)."""
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("not found")
        mock_client.get_or_create_collection.return_value = mock_col

        with patch("chromadb.PersistentClient", return_value=mock_client):
            from tools.recipe import init_chroma
            init_chroma(str(tmp_path / "chroma"))

        mock_col.add.assert_called_once()

    def test_second_init_no_reseed(self, tmp_path):
        """init_chroma with non-empty collection does NOT call add."""
        mock_col = MagicMock()
        mock_col.count.return_value = 10
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("not found")
        mock_client.get_or_create_collection.return_value = mock_col

        with patch("chromadb.PersistentClient", return_value=mock_client):
            from tools.recipe import init_chroma
            init_chroma(str(tmp_path / "chroma"))

        mock_col.add.assert_not_called()


# ── Conversation Memory ────────────────────────────────────────────────────────


class TestConversationMemory:
    def test_empty_collection_is_valid(self, monkeypatch, tmp_path):
        """init_conversation_memory_chroma creates the collection without seeding."""
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        mock_client = MagicMock()
        mock_col = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_col

        with patch("chromadb.PersistentClient", return_value=mock_client):
            from tools.memory import init_conversation_memory_chroma
            init_conversation_memory_chroma()

        mock_client.get_or_create_collection.assert_called_once()
        mock_col.delete.assert_not_called()

    def test_second_init_no_delete(self, monkeypatch, tmp_path):
        """Calling init_conversation_memory_chroma twice never deletes the collection."""
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        mock_client = MagicMock()
        mock_col = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_col

        with patch("chromadb.PersistentClient", return_value=mock_client):
            from tools.memory import init_conversation_memory_chroma
            init_conversation_memory_chroma()
            init_conversation_memory_chroma()

        mock_client.delete_collection.assert_not_called()

    def test_uses_chroma_persist_dir(self, monkeypatch, tmp_path):
        """init_conversation_memory_chroma reads path from CHROMA_PERSIST_DIR."""
        custom = str(tmp_path / "conv_chroma")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", custom)

        with patch("chromadb.PersistentClient") as mock_ctor:
            mock_ctor.return_value.get_or_create_collection.return_value = MagicMock()
            from tools.memory import init_conversation_memory_chroma
            init_conversation_memory_chroma()

        mock_ctor.assert_called_with(path=custom)


# ── Scheduler ─────────────────────────────────────────────────────────────────


class TestEnvFlag:
    @pytest.mark.parametrize("val", ["true", "True", "TRUE", "1", "yes", "YES", "on", "ON"])
    def test_true_values(self, monkeypatch, val):
        """All truthy string variants are parsed as True."""
        monkeypatch.setenv("_TEST_FLAG_10D2", val)
        from core.runtime_paths import env_flag
        assert env_flag("_TEST_FLAG_10D2") is True

    @pytest.mark.parametrize("val", ["false", "False", "FALSE", "0", "no", "NO", "off", "OFF"])
    def test_false_values(self, monkeypatch, val):
        """All falsy string variants are parsed as False."""
        monkeypatch.setenv("_TEST_FLAG_10D2", val)
        from core.runtime_paths import env_flag
        assert env_flag("_TEST_FLAG_10D2") is False

    def test_invalid_value_raises(self, monkeypatch):
        """Unknown value raises ValueError — not silently treated as true or false."""
        monkeypatch.setenv("_TEST_FLAG_10D2", "maybe")
        from core.runtime_paths import env_flag
        with pytest.raises(ValueError, match="_TEST_FLAG_10D2"):
            env_flag("_TEST_FLAG_10D2")


_LIFESPAN_PATCHES = [
    "core.runtime_paths.ensure_runtime_dirs",
    "main.init_chroma",
    "main.init_nutrition_chroma",
    "main.init_food_safety_chroma",
    "main.init_conversation_memory_chroma",
    "main.load_nutrition_documents",
    "main.load_food_safety_documents",
    "main.init_db",
    "main.init_dead_letter_db",
    "main.start_scheduler",
    "main.stop_scheduler",
]


def _enter_lifespan_patches(stack: ExitStack, extras: dict | None = None) -> dict:
    """Enter all standard lifespan patches, return dict of name→mock."""
    mocks = {}
    for target in _LIFESPAN_PATCHES:
        mocks[target] = stack.enter_context(patch(target))
    if extras:
        for target, kwargs in extras.items():
            mocks[target] = stack.enter_context(patch(target, **kwargs))
    return mocks


class TestSchedulerLifecycle:
    @pytest.mark.asyncio
    async def test_run_scheduler_false_no_start_stop(self, monkeypatch, tmp_path):
        """RUN_SCHEDULER=false: start_scheduler and stop_scheduler are never called."""
        monkeypatch.setenv("RUN_SCHEDULER", "false")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))

        with ExitStack() as stack:
            mocks = _enter_lifespan_patches(stack)
            import main as m
            async with m.lifespan(m.app):
                pass

        mocks["main.start_scheduler"].assert_not_called()
        mocks["main.stop_scheduler"].assert_not_called()

    @pytest.mark.asyncio
    async def test_run_scheduler_true_starts_and_stops(self, monkeypatch, tmp_path):
        """RUN_SCHEDULER=true: start_scheduler once on enter, stop_scheduler once on exit."""
        monkeypatch.setenv("RUN_SCHEDULER", "true")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))

        with ExitStack() as stack:
            mocks = _enter_lifespan_patches(stack)
            import main as m
            async with m.lifespan(m.app):
                pass

        assert mocks["main.start_scheduler"].call_count == 1
        assert mocks["main.stop_scheduler"].call_count == 1

    @pytest.mark.asyncio
    async def test_lifespan_body_exception_still_stops(self, monkeypatch, tmp_path):
        """An exception raised inside the lifespan body still triggers stop_scheduler."""
        monkeypatch.delenv("RUN_SCHEDULER", raising=False)  # default True
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))

        with ExitStack() as stack:
            mocks = _enter_lifespan_patches(stack)
            import main as m
            with pytest.raises(RuntimeError, match="body error"):
                async with m.lifespan(m.app):
                    raise RuntimeError("body error")

        mocks["main.stop_scheduler"].assert_called_once()

    @pytest.mark.asyncio
    async def test_init_fails_before_scheduler_no_stop(self, monkeypatch, tmp_path):
        """If init_chroma raises before scheduler starts, stop_scheduler is NOT called."""
        monkeypatch.delenv("RUN_SCHEDULER", raising=False)
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))

        with (
            patch("core.runtime_paths.ensure_runtime_dirs"),
            patch("main.init_db"),
            patch("main.init_dead_letter_db"),
            patch("main.init_chroma", side_effect=RuntimeError("chroma failed")),
            patch("main.start_scheduler") as mock_start,
            patch("main.stop_scheduler") as mock_stop,
        ):
            import main as m
            with pytest.raises(RuntimeError, match="chroma failed"):
                async with m.lifespan(m.app):
                    pass  # never reached

        mock_start.assert_not_called()
        mock_stop.assert_not_called()

    def test_start_scheduler_idempotent(self):
        """Calling start_scheduler when scheduler.running=True returns without re-starting."""
        import agents.maintenance_agent as ma
        mock_sched = MagicMock()
        mock_sched.running = True

        with patch.object(ma, "scheduler", mock_sched):
            ma.start_scheduler()

        mock_sched.start.assert_not_called()
        mock_sched.add_job.assert_not_called()

    def test_stop_scheduler_idempotent(self):
        """Calling stop_scheduler when scheduler.running=False is a no-op."""
        import agents.maintenance_agent as ma
        mock_sched = MagicMock()
        mock_sched.running = False

        with patch.object(ma, "scheduler", mock_sched):
            ma.stop_scheduler()
            ma.stop_scheduler()

        mock_sched.shutdown.assert_not_called()

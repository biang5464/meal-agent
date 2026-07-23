"""Phase 10D5A: embedding runtime singleton tests.

Verifies that recipe, nutrition, and food_safety share one BGE model instance,
and that _cosine_similarity no longer creates a new SentenceTransformer per call.
All tests avoid downloading the real model by injecting a FakeEF.
"""
from __future__ import annotations

import inspect
import threading
import time
from unittest.mock import patch

import pytest


# ---------- helpers ----------

class _FakeEF:
    """Deterministic fake embedding function — no model download needed."""
    def __init__(self):
        self.call_count = 0

    def __call__(self, texts):
        self.call_count += 1
        out = []
        for t in texts:
            # one-hot vector of length 16: index = hash(t) % 16
            v = [0.0] * 16
            v[hash(t) % 16] = 1.0
            out.append(v)
        return out


@pytest.fixture(autouse=True)
def _reset_singleton():
    from core import embedding_runtime
    embedding_runtime.reset_embedding_runtime_for_tests()
    yield
    embedding_runtime.reset_embedding_runtime_for_tests()


@pytest.fixture
def fake_ef():
    """Inject a FakeEF into the singleton so tests never touch the real model."""
    from core import embedding_runtime
    ef = _FakeEF()
    embedding_runtime._ef = ef
    return ef


# ---------- singleton behaviour ----------

class TestSingleton:
    def test_not_loaded_at_import(self):
        from core import embedding_runtime
        # After reset, singleton must be None (lazy load, not import-time)
        assert embedding_runtime._ef is None

    def test_same_object_on_repeated_calls(self, fake_ef):
        from core.embedding_runtime import get_embedding_function
        a = get_embedding_function()
        b = get_embedding_function()
        assert a is b

    def test_returns_injected_fake(self, fake_ef):
        from core.embedding_runtime import get_embedding_function
        assert get_embedding_function() is fake_ef

    def test_reset_clears_singleton(self, fake_ef):
        from core import embedding_runtime
        assert embedding_runtime._ef is not None
        embedding_runtime.reset_embedding_runtime_for_tests()
        assert embedding_runtime._ef is None

    def test_concurrent_calls_load_once(self, monkeypatch):
        """Multiple threads racing get_embedding_function() create exactly one instance."""
        from core import embedding_runtime

        load_count = 0

        class _CountingEF:
            def __call__(self, texts):
                return [[] for _ in texts]

        def _fake_sef_cls(model_name):
            nonlocal load_count
            time.sleep(0.02)  # simulate slow model load
            load_count += 1
            return _CountingEF()

        monkeypatch.setattr(
            "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
            _fake_sef_cls,
        )

        results = []

        def _get():
            results.append(embedding_runtime.get_embedding_function())

        threads = [threading.Thread(target=_get) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert load_count == 1, f"Model was created {load_count} times; expected 1"
        ids = {id(r) for r in results}
        assert len(ids) == 1, "Threads received different EF instances"


# ---------- shared across collections ----------

class TestSharedAcrossCollections:
    def test_recipe_make_ef_returns_singleton(self, fake_ef):
        from tools.recipe import _make_ef
        assert _make_ef() is fake_ef

    def test_nutrition_make_ef_returns_singleton(self, fake_ef):
        from tools.nutrition import _make_ef
        assert _make_ef() is fake_ef

    def test_recipe_and_nutrition_share_same_ef(self, fake_ef):
        from tools.recipe import _make_ef as recipe_ef
        from tools.nutrition import _make_ef as nutrition_ef
        assert recipe_ef() is nutrition_ef()

    def test_no_local_ef_class_import_in_recipe(self):
        """recipe.py must not import SentenceTransformerEmbeddingFunction at top level."""
        import tools.recipe as r_mod
        src = inspect.getsource(r_mod)
        assert "from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction" not in src

    def test_no_local_ef_class_import_in_nutrition(self):
        """nutrition.py must not import SentenceTransformerEmbeddingFunction at top level."""
        import tools.nutrition as n_mod
        src = inspect.getsource(n_mod)
        assert "from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction" not in src


# ---------- cosine similarity ----------

class TestCosineSimilarity:
    def test_no_direct_sentence_transformer_import(self):
        """_cosine_similarity must not instantiate SentenceTransformer directly."""
        from agents.context_manager import _cosine_similarity
        src = inspect.getsource(_cosine_similarity)
        assert "SentenceTransformer(" not in src, (
            "_cosine_similarity still creates a SentenceTransformer instance"
        )

    def test_uses_embed_texts(self):
        from agents.context_manager import _cosine_similarity
        src = inspect.getsource(_cosine_similarity)
        assert "embed_texts" in src

    def test_identical_texts_cosine_approx_one(self, fake_ef):
        from agents.context_manager import _cosine_similarity
        result = _cosine_similarity("宫保鸡丁", "宫保鸡丁")
        assert abs(result - 1.0) < 1e-6, f"cosine(x,x) should be ~1.0, got {result}"

    def test_orthogonal_texts_cosine_zero(self, fake_ef):
        """Two texts with no hash-slot overlap → orthogonal vectors → cosine 0."""
        from agents.context_manager import _cosine_similarity
        seen = set()
        texts = []
        for candidate in ["apple", "orange", "banana", "grape", "melon", "berry",
                           "peach", "plum", "kiwi", "mango", "papaya", "lemon"]:
            slot = hash(candidate) % 16
            if slot not in seen:
                seen.add(slot)
                texts.append(candidate)
            if len(texts) == 2:
                break

        if len(texts) < 2:
            pytest.skip("could not find two non-colliding hash slots")

        result = _cosine_similarity(texts[0], texts[1])
        assert result == pytest.approx(0.0, abs=1e-6), (
            f"Orthogonal texts should have cosine ~0, got {result}"
        )

    def test_failure_fallback_returns_zero(self, monkeypatch):
        """If embed_texts raises, _cosine_similarity degrades to 0.0."""
        monkeypatch.setattr(
            "core.embedding_runtime.embed_texts",
            lambda texts: (_ for _ in ()).throw(RuntimeError("model error")),
        )
        from agents.context_manager import _cosine_similarity
        result = _cosine_similarity("a", "b")
        assert result == 0.0


# ---------- embed_texts ----------

class TestEmbedTexts:
    def test_returns_list_of_lists(self, fake_ef):
        from core.embedding_runtime import embed_texts
        result = embed_texts(["宫保鸡丁", "番茄炒蛋"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(v, list) for v in result)

    def test_failure_returns_empty_inner_lists(self, monkeypatch):
        from core import embedding_runtime
        embedding_runtime.reset_embedding_runtime_for_tests()

        class _BrokenEF:
            def __call__(self, texts):
                raise RuntimeError("broken")

        embedding_runtime._ef = _BrokenEF()
        result = embedding_runtime.embed_texts(["a", "b"])
        assert result == [[], []]


# ---------- numpy EF compatibility (hotfix regression guard) ----------

class TestNumpyEFCompatibility:
    """Guard against the root-cause regression: EF returns np.ndarray, not list.

    The real SentenceTransformerEmbeddingFunction returns numpy arrays.
    embed_texts must convert them so downstream `if not vectors[i]` never hits
    the 'truth value of an array is ambiguous' ValueError.
    """

    @pytest.fixture
    def numpy_ef(self):
        import numpy as np
        from core import embedding_runtime

        class _NumpyEF:
            """Returns np.ndarray per text, matching real EF output shape."""
            def __call__(self, texts):
                return [np.array([float(i == hash(t) % 4) for i in range(4)]) for t in texts]

        ef = _NumpyEF()
        embedding_runtime._ef = ef
        return ef

    def test_embed_texts_converts_numpy_to_plain_list(self, numpy_ef):
        """embed_texts must return list[list[...]] even when EF returns np.ndarray."""
        from core.embedding_runtime import embed_texts
        result = embed_texts(["苹果", "香蕉"])
        assert isinstance(result, list)
        assert all(isinstance(v, list) for v in result), (
            "embed_texts must convert numpy arrays to plain Python lists"
        )

    def test_embed_texts_elements_support_python_truthiness(self, numpy_ef):
        """Plain `not result[i]` must not raise ValueError (no numpy ambiguity)."""
        from core.embedding_runtime import embed_texts
        result = embed_texts(["苹果"])
        _ = not result[0]   # would raise ValueError if result[0] is np.ndarray
        _ = sum(result[0])  # plain Python numeric operations must work

    def test_cosine_no_silent_zero_with_numpy_ef(self, numpy_ef):
        """_cosine_similarity must NOT silently degrade to 0.0 when EF returns numpy arrays."""
        from agents.context_manager import _cosine_similarity
        result = _cosine_similarity("苹果", "苹果")
        assert result == pytest.approx(1.0, abs=1e-6), (
            f"cosine(x, x) with numpy EF returned {result}; "
            "likely numpy truth-value ValueError silently caught → 0.0"
        )


# ---------- collection names unchanged ----------

class TestCollectionNamesUnchanged:
    def test_recipe_collection_name(self):
        import pathlib
        src = (pathlib.Path(__file__).parent.parent / "tools" / "recipe.py").read_text(encoding="utf-8")
        assert '"recipes"' in src or "'recipes'" in src

    def test_nutrition_collection_name(self):
        from tools.nutrition import NUTRITION_COLLECTION
        assert NUTRITION_COLLECTION == "nutrition"

    def test_food_safety_collection_name(self):
        from tools.nutrition import FOOD_SAFETY_COLLECTION
        assert FOOD_SAFETY_COLLECTION == "food_safety"

    def test_conversation_memory_not_in_embedding_runtime(self):
        """conversation_memory must NOT appear in embedding_runtime.py."""
        from core import embedding_runtime
        src = inspect.getsource(embedding_runtime)
        assert "conversation_memory" not in src

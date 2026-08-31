from app.services.reranker_providers.qwen3_reranker_provider import Qwen3RerankerProvider


class FakeCrossEncoder:
    def __init__(self):
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return [0.2, 0.9]


def test_qwen_reranker_is_lazy_cached_and_orders_scores_without_network():
    Qwen3RerankerProvider._models.clear()
    loads = []

    def loader(model_id, device, instruction):
        loads.append((model_id, device, instruction))
        return FakeCrossEncoder()

    provider = Qwen3RerankerProvider(model_id="test-reranker", device="cpu", loader=loader)
    assert loads == []
    scores = provider.rerank("français العربية", ["premier", "الثاني"], 2)
    assert loads[0][:2] == ("test-reranker", "cpu")
    assert "pedagogical query" in loads[0][2]
    assert [(item.index, item.score) for item in scores] == [(1, 0.9), (0, 0.2)]
    provider.rerank("q", ["a", "b"], 1)
    assert len(loads) == 1


def test_qwen_loader_overrides_the_generic_web_search_prompt(monkeypatch):
    captured = {}

    class FakeSentenceTransformers:
        class CrossEncoder:
            def __init__(self, model_id, **kwargs):
                captured["model_id"] = model_id
                captured.update(kwargs)

    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", FakeSentenceTransformers)
    instruction = "Given a pedagogical query, prefer explicit lesson goals."
    Qwen3RerankerProvider._default_loader("test-reranker", "cpu", instruction)
    assert captured["prompts"] == {"pedagogical": instruction}
    assert captured["default_prompt_name"] == "pedagogical"
    assert "web search query" not in captured["prompts"]["pedagogical"]

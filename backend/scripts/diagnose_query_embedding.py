"""Read-only isolation diagnostic for the production Qwen query embedding path."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))
SCENARIO_C_QUERY = (
    "enseignement de l'arabe; activité pédagogique; niveau A2; "
    "contexte pédagogique: Current request:\n"
    "Maintenant adapte-la au niveau A2 et travaille la compréhension orale.\n\n"
    "Recent pedagogical context:\n"
    "Propose-moi une sur la famille.\n"
    "Rends-la plus facile et limite-la à 10 minutes.; compétences: "
    "compréhension orale, écoute, activité d'écoute"
)
A1_QUERY = (
    "enseignement de l'arabe; activité pédagogique; niveau A1; "
    "thème: la famille; compétences: expression orale, interaction orale, production orale"
)


def _memory() -> dict[str, int | None]:
    result: dict[str, int | None] = {"rss_bytes": None, "available_ram_bytes": None}
    try:
        import psutil

        result["rss_bytes"] = psutil.Process(os.getpid()).memory_info().rss
        result["available_ram_bytes"] = psutil.virtual_memory().available
    except Exception:
        pass
    return result


def _emit(event: str, **details: object) -> None:
    print(json.dumps({"event": event, **details}, ensure_ascii=False), flush=True)


def _provider_details(provider, model) -> dict[str, object]:
    import torch

    tokenizer = getattr(model, "tokenizer", None)
    return {
        "provider_class": type(provider).__name__,
        "model_id": provider.model_id,
        "configured_device": provider.device,
        "selected_device": provider._resolved_device(),
        "model_class": type(model).__name__,
        "max_sequence_length": getattr(model, "max_seq_length", None),
        "tokenizer_max_length": getattr(tokenizer, "model_max_length", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_allocated_bytes": torch.cuda.memory_allocated() if torch.cuda.is_available() else None,
        "cuda_reserved_bytes": torch.cuda.memory_reserved() if torch.cuda.is_available() else None,
        "query_instruction": provider.query_instruction,
    }


def _load_provider():
    from app.services.embedding_providers import get_embedding_provider

    provider = get_embedding_provider()
    _emit("model_load_started", memory=_memory())
    started = time.perf_counter()
    model = provider._get_model()
    _emit(
        "model_load_completed",
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        memory=_memory(),
        **_provider_details(provider, model),
    )
    return provider, model


def _encode_query(provider, model, query: str, label: str) -> None:
    tokenizer = getattr(model, "tokenizer", None)
    token_count = None
    if tokenizer is not None:
        token_count = len(tokenizer.encode(query, add_special_tokens=True))
    _emit(
        "embedding_query_started",
        label=label,
        character_length=len(query),
        token_count=token_count,
        memory=_memory(),
    )
    started = time.perf_counter()
    vector = provider.embed_queries([query])[0]
    _emit(
        "embedding_query_completed",
        label=label,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        vector_dimension=len(vector),
        finite=all(math.isfinite(value) for value in vector),
        memory=_memory(),
    )


def _child(mode: str) -> None:
    provider, model = _load_provider()
    if mode == "load":
        return
    if mode == "short":
        _encode_query(provider, model, "famille", "famille")
        return
    if mode == "a1":
        _encode_query(provider, model, A1_QUERY, "a1_pedagogical")
        return
    if mode == "scenario_c":
        _encode_query(provider, model, SCENARIO_C_QUERY, "scenario_c")
        return
    if mode == "reuse":
        for query in ("famille", "école", "compréhension orale"):
            _encode_query(provider, model, query, query)
        return
    raise ValueError(f"Unsupported diagnostic mode: {mode}")


def _run_child(mode: str, timeout_seconds: float) -> dict[str, object]:
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--child", mode]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        return {
            "mode": mode,
            "result": "completed",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "exit_code": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "mode": mode,
            "result": "timeout",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "exit_code": None,
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "").strip() if isinstance(exc.stderr, str) else "",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=("load", "short", "a1", "scenario_c", "reuse"))
    parser.add_argument("--case", choices=("load", "short", "a1", "scenario_c", "reuse"))
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if args.child:
        _child(args.child)
        return

    _emit("diagnostic_started", timeout_seconds=args.timeout_seconds, memory=_memory())
    modes = (args.case,) if args.case else ("load", "short", "a1", "scenario_c", "reuse")
    for mode in modes:
        _emit("diagnostic_case", **_run_child(mode, args.timeout_seconds))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tabulate import tabulate

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return proportion of expected facts present in answer (0.0 to 1.0)."""
    if not expected:
        return 1.0
    # Word-boundary match so short tokens such as "AI" cannot be hit by accident
    # inside unrelated words (e.g. "hai", "mai").
    # Acronyms (all caps, e.g. "AI") are matched case-sensitively: the Vietnamese
    # pronoun "ai" ("who") must not count as a hit.
    def hit(exp: str) -> bool:
        flags = 0 if exp.isupper() else re.IGNORECASE
        return re.search(rf"(?<!\w){re.escape(exp)}(?!\w)", answer, flags) is not None

    matches = sum(1 for exp in expected if hit(exp))
    return matches / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Evaluate response quality based on recall, conciseness, and structure."""
    if not answer.strip():
        return 0.0

    rec = recall_points(answer, expected)
    score = rec * 0.7

    # Bonus for bullet point structuring
    if "-" in answer or "\n" in answer:
        score += 0.15

    # Bonus for appropriate length (concise yet informative)
    length = len(answer)
    if 30 <= length <= 350:
        score += 0.15
    elif length > 0:
        score += 0.05

    return round(min(1.0, score), 3)


def run_agent_benchmark(
    agent_name: str,
    agent: Any,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Evaluate one agent across multiple conversations and fresh recall threads."""
    all_recall_scores: list[float] = []
    all_quality_scores: list[float] = []
    all_threads: list[str] = []
    user_ids: set[str] = set()

    for conv in conversations:
        conv_id = conv["id"]
        user_id = conv["user_id"]
        user_ids.add(user_id)
        turns: list[str] = conv.get("turns", [])
        recall_questions: list[dict[str, Any]] = conv.get("recall_questions", [])

        # 1. Main conversation thread
        main_thread = f"{conv_id}_main"
        all_threads.append(main_thread)
        for turn in turns:
            agent.reply(user_id, main_thread, turn)

        # 2. Recall evaluation in a fresh thread
        for idx, q_item in enumerate(recall_questions):
            recall_thread = f"{conv_id}_recall_{idx}"
            all_threads.append(recall_thread)
            question = q_item["question"]
            expected = q_item["expected_contains"]

            res = agent.reply(user_id, recall_thread, question)
            ans_text = res.get("content", "")

            r_pts = recall_points(ans_text, expected)
            q_pts = heuristic_quality(ans_text, expected)

            all_recall_scores.append(r_pts)
            all_quality_scores.append(q_pts)

    # Aggregate token metrics across all threads
    total_agent_tokens = sum(agent.token_usage(t) for t in all_threads)
    total_prompt_tokens = sum(agent.prompt_token_usage(t) for t in all_threads)
    total_compactions = sum(agent.compaction_count(t) for t in all_threads)

    # Compute memory file growth
    total_memory_growth = 0
    if hasattr(agent, "memory_file_size"):
        total_memory_growth = sum(agent.memory_file_size(u) for u in user_ids)

    avg_recall = (sum(all_recall_scores) / len(all_recall_scores)) if all_recall_scores else 0.0
    avg_quality = (sum(all_quality_scores) / len(all_quality_scores)) if all_quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 3),
        response_quality=round(avg_quality, 3),
        memory_growth_bytes=total_memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a markdown table."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = []
    for r in rows:
        table_data.append([
            r.agent_name,
            f"{r.agent_tokens_only:,}",
            f"{r.prompt_tokens_processed:,}",
            f"{r.recall_score * 100:.1f}%",
            f"{r.response_quality * 100:.1f}%",
            f"{r.memory_growth_bytes:,} B",
            r.compactions,
        ])
    return tabulate(table_data, headers=headers, tablefmt="github")


def _fresh_config(config: LabConfig, root: Path) -> LabConfig:
    """Config whose state dir is empty, so every benchmark run starts from zero memory."""
    state = root / "state"
    (state / "profiles").mkdir(parents=True, exist_ok=True)
    return replace(config, state_dir=state)


def run_pair(config: LabConfig, convs: list[dict[str, Any]], **advanced_kwargs: Any) -> tuple[BenchmarkRow, BenchmarkRow]:
    """Run baseline and advanced on identical input with isolated, empty state."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _fresh_config(config, Path(tmp))
        base = run_agent_benchmark("Baseline Agent", BaselineAgent(cfg, force_offline=True), convs, cfg)
        adv = run_agent_benchmark("Advanced Agent", AdvancedAgent(cfg, force_offline=True, **advanced_kwargs), convs, cfg)
    return base, adv


def length_sweep(config: LabConfig, conv: dict[str, Any], lengths: list[int]) -> str:
    """Prompt tokens vs conversation length: where does compact start to pay off?"""
    rows = []
    for n in lengths:
        sub = {**conv, "turns": conv["turns"][:n], "recall_questions": []}
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _fresh_config(config, Path(tmp))
            base, adv = BaselineAgent(cfg, force_offline=True), AdvancedAgent(cfg, force_offline=True)
            for t in sub["turns"]:
                base.reply(sub["user_id"], "sweep", t)
                adv.reply(sub["user_id"], "sweep", t)
            b, a = base.prompt_token_usage("sweep"), adv.prompt_token_usage("sweep")
            rows.append([n, f"{b:,}", f"{a:,}", f"{(a - b) / b * 100:+.0f}%", adv.compaction_count("sweep"),
                         "Advanced rẻ hơn" if a < b else "Baseline rẻ hơn"])
    return tabulate(rows, headers=["Số lượt", "Baseline prompt", "Advanced prompt", "Chênh lệch", "Compactions", "Rẻ hơn"], tablefmt="github")


def ablation(config: LabConfig, convs: list[dict[str, Any]]) -> str:
    """Effect of the confidence threshold (noise gate) on recall and rejected facts."""
    rows = []
    for label, thr in (("Không có gate (threshold=0.0)", 0.0), ("Có gate (threshold=0.6)", 0.6)):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _fresh_config(config, Path(tmp))
            agent = AdvancedAgent(cfg, force_offline=True, confidence_threshold=thr)
            row = run_agent_benchmark("Advanced Agent", agent, convs, cfg)
            rows.append([label, f"{row.recall_score * 100:.1f}%", agent.rejected_candidates, len(agent.corrections)])
    return tabulate(rows, headers=["Cấu hình", "Cross-session recall", "Candidate bị từ chối", "Số lần correction"], tablefmt="github")


def main() -> None:
    """Run Standard and Long-Context Stress benchmarks comparing Baseline and Advanced."""
    base_dir = Path(__file__).resolve().parent.parent
    config = load_config(base_dir)
    standard_convs = load_conversations(config.data_dir / "conversations.json")
    stress_convs = load_conversations(config.data_dir / "advanced_long_context.json")

    sections: list[tuple[str, str]] = []
    sections.append(("Standard Benchmark (data/conversations.json)", format_rows(list(run_pair(config, standard_convs)))))
    sections.append(("Long-Context Stress Benchmark (data/advanced_long_context.json)", format_rows(list(run_pair(config, stress_convs)))))
    sections.append(("Length sweep - hội thoại ngắn (conv-01, prefix 1..10 lượt)", length_sweep(config, standard_convs[0], [1, 2, 4, 6, 8, 10])))
    sections.append(("Length sweep - hội thoại dài (stress-01, prefix 1..16 lượt)", length_sweep(config, stress_convs[0], [1, 2, 4, 6, 8, 12, 16])))
    sections.append(("Ablation - Confidence threshold (Standard)", ablation(config, standard_convs)))
    sections.append(("Ablation - Confidence threshold (Stress)", ablation(config, stress_convs)))

    out: list[str] = []
    for title, body in sections:
        block = f"\n{'=' * 60}\n{title}\n{'=' * 60}\n{body}"
        print(block)
        out.append(f"## {title}\n\n{body}\n")

    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "benchmark.md").write_text("# Benchmark results (offline, deterministic)\n\n" + "\n".join(out), encoding="utf-8")
    print(f"\nSaved to {results_dir / 'benchmark.md'}")


if __name__ == "__main__":
    main()

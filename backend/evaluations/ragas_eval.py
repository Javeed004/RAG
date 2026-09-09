import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import requests

# CONFIG

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 60
EMBEDDING_MODEL = os.environ.get("RAGAS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Below these thresholds a metric is flagged as "low" in the report.
FAITHFULNESS_THRESHOLD = 0.7
RELEVANCY_THRESHOLD = 0.7
CONTEXT_PRECISION_THRESHOLD = 0.7
CONTEXT_RECALL_THRESHOLD = 0.7


@dataclass
class QAPair:
    id: str
    source_doc: str
    question: str
    reference: str  # the correct, verified answer (ground truth)


# THE 13-ITEM TEST SET (>= 10 required) — built from documents actually
# ingested into the corpus, with answers verified against the source text.

QA_DATASET: List[QAPair] = [
    QAPair(
        id="EVO-1",
        source_doc="Evolution (Kutschera, 2013)",
        question="Who defined biological evolution as 'descent with modification', and in which work?",
        reference="Charles Darwin defined it as 'descent with modification' in his book "
                   "On the Origin of Species, published in 1859.",
    ),
    QAPair(
        id="EVO-2",
        source_doc="Evolution (Kutschera, 2013)",
        question="How does the article's glossary define 'Darwinian fitness'?",
        reference="Darwinian fitness is defined as the relative lifetime reproductive success "
                   "of an individual within a population, i.e. the number of surviving offspring.",
    ),
    QAPair(
        id="EVO-3",
        source_doc="Evolution (Kutschera, 2013)",
        question="What are the two geographic categories of speciation the article distinguishes?",
        reference="The article distinguishes allopatric speciation, where two populations' ranges "
                   "do not overlap, from sympatric speciation, where their ranges overlap.",
    ),
    QAPair(
        id="FUN-1",
        source_doc="Funnel Analysis",
        question="How many unique users reached the Visited Site stage of the funnel?",
        reference="200 unique users reached the Visited Site stage, the funnel's entry point.",
    ),
    QAPair(
        id="FUN-2",
        source_doc="Funnel Analysis",
        question="What was the overall conversion rate from Visited Site to Purchase Completed?",
        reference="The overall conversion rate from Visited Site to Purchase Completed was 22.0%.",
    ),
    QAPair(
        id="FUN-3",
        source_doc="Funnel Analysis",
        question="Which stage-to-stage transition had the biggest drop-off, and what was the rate?",
        reference="The biggest drop-off was between Signup Started and Details Filled, at a "
                   "36.0% drop-off rate, losing 54 users.",
    ),
    QAPair(
        id="MAG-1",
        source_doc="mAgeing Implementation Handbook",
        question="What are the two streams of the mAgeing programme described in the handbook?",
        reference="The two streams are mAgeing ICOPE (used alongside a health care worker's care "
                   "plan) and mAgeing (a standalone stream for those without a care plan).",
    ),
    QAPair(
        id="MAG-2",
        source_doc="mAgeing Implementation Handbook",
        question="Of the 6 ICOPE actions to manage declines in intrinsic capacity, how many does "
                 "the mAgeing programme include, and which one is left out?",
        reference="The mAgeing programme includes 5 of the 6 ICOPE actions; it excludes action "
                   "6, supporting caregivers.",
    ),
    QAPair(
        id="MAG-3",
        source_doc="mAgeing Implementation Handbook",
        question="According to the handbook's recommendations, how long should an mAgeing programme last, at minimum?",
        reference="The mAgeing programme should last at least six months (24 weeks).",
    ),
    QAPair(
        id="LT-1",
        source_doc="L&T Interview Invitation Letter",
        question="What refundable fee amount does the L&T interview letter say candidates must pay to confirm participation?",
        reference="The letter states candidates must pay a refundable fee of INR 1,850 "
                   "(including GST) to confirm their participation.",
    ),
]


# STEP 1 — drive the real /chat endpoint to get answer + retrieved contexts

def call_chat(base_url: str, question: str, timeout: int) -> dict:
    url = base_url.rstrip("/") + "/chat"
    try:
        resp = requests.post(url, json={"question": question, "history": []}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        return {"_error": str(exc)}


def build_eval_rows(base_url: str, timeout: int) -> List[dict]:
    rows = []
    for qa in QA_DATASET:
        print(f"[{qa.id}] querying: {qa.question}")
        start = time.time()
        resp = call_chat(base_url, qa.question, timeout)
        elapsed = time.time() - start

        if "_error" in resp:
            print(f"    -> ERROR: {resp['_error']}")
            rows.append({
                "qa": qa,
                "answer": "",
                "contexts": [],
                "elapsed": elapsed,
                "error": resp["_error"],
            })
            continue

        answer = resp.get("answer", "")
        sources = resp.get("sources", [])
        contexts = [s.get("chunk", "") for s in sources if s.get("chunk")]

        print(f"    -> {elapsed:.1f}s, {len(contexts)} context chunks retrieved")
        rows.append({
            "qa": qa,
            "answer": answer,
            "contexts": contexts,
            "elapsed": elapsed,
            "error": None,
        })
    return rows


# STEP 2 — run RAGAS

def run_ragas(rows: List[dict]):
    """
    Returns (per_row_scores: list[dict], ragas_result) or raises with a
    clear message if ragas / an evaluator LLM isn't available.
    """
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.run_config import RunConfig
    except ImportError as exc:
        raise SystemExit(
            "RAGAS (or a dependency) isn't installed. Run:\n"
            "    pip install ragas datasets langchain-huggingface\n"
            f"Original error: {exc}"
        )

    # --- evaluator LLM: reuse your own project's LLM factory ------------
    PROJECT_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    sys.path.insert(0, PROJECT_ROOT)

    try:
        from llm.llm_factory import get_llm

        base_llm = get_llm()

    except Exception as exc:
        raise SystemExit(
            "Could not import get_llm() from llm.llm_factory.\n"
            f"Backend path used: {PROJECT_ROOT}\n"
            f"Original error: {exc}"
        )
    try:
        from llm.llm_factory import get_llm  
        base_llm = get_llm()
    except Exception as exc:  
        raise SystemExit(
            "Could not import get_llm() from llm.llm_factory. Run this script "
            "from your RAG-Project root (or adjust the sys.path insert above "
            f"to point at your backend/ folder). Original error: {exc}"
        )
    run_config = RunConfig(
        timeout=300,
        max_retries=1,
        max_workers=1,
    )

    evaluator_llm = LangchainLLMWrapper(
        base_llm,
        bypass_n=True,
        run_config=run_config,
    )

    # --- evaluator embeddings --------------------------------------------
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # fallback
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )

    # --- build the RAGAS dataset ------------------------------------------
    dataset_rows = []
    usable_rows = [r for r in rows if r["error"] is None]
    for r in usable_rows:
        dataset_rows.append({
            "user_input": r["qa"].question,
            "response": r["answer"],
            "retrieved_contexts": r["contexts"] or [""],  # ragas needs a non-empty list
            "reference": r["qa"].reference,
        })

    if not dataset_rows:
        raise SystemExit("No successful /chat responses to evaluate — check the backend is running.")

    eval_dataset = EvaluationDataset.from_list(dataset_rows)

    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]

    print("\nRunning RAGAS evaluation (this calls the evaluator LLM several times per row)...\n")
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    scores_df = result.to_pandas()
    per_row = scores_df.to_dict(orient="records")

    # re-attach the ids / source_doc / retrieval error info that ragas doesn't carry
    for row, usable in zip(per_row, usable_rows):
        row["id"] = usable["qa"].id
        row["source_doc"] = usable["qa"].source_doc
        row["elapsed"] = usable["elapsed"]

    return per_row, result


# ---------------------------------------------------------------------------
# STEP 3 — root-cause each low-scoring row
# ---------------------------------------------------------------------------

def diagnose(row: dict) -> str:
    faithfulness = row.get("faithfulness")
    relevancy = row.get("answer_relevancy")
    precision = row.get("context_precision")
    recall = row.get("context_recall")

    problems = []
    retrieval_bad = (precision is not None and precision < CONTEXT_PRECISION_THRESHOLD) or \
                     (recall is not None and recall < CONTEXT_RECALL_THRESHOLD)
    faithfulness_bad = faithfulness is not None and faithfulness < FAITHFULNESS_THRESHOLD
    relevancy_bad = relevancy is not None and relevancy < RELEVANCY_THRESHOLD

    if retrieval_bad:
        problems.append(
            "RETRIEVAL: low context_precision/recall — the retriever likely fetched the "
            "wrong chunks or missed the chunk containing the answer. Check chunking size, "
            "overlap, or the embedding model's similarity threshold for this query."
        )
    if faithfulness_bad and not retrieval_bad:
        problems.append(
            "GENERATION: context looks adequate but faithfulness is low — the LLM is "
            "adding claims not supported by the retrieved context. Tighten the grounding "
            "rules in the system prompt (see prompts.py rule #1/#7)."
        )
    if faithfulness_bad and retrieval_bad:
        problems.append(
            "COMPOUND: faithfulness is low AND retrieval is weak — the low faithfulness is "
            "likely a downstream symptom of bad retrieval (the model had nothing good to "
            "ground on). Fix retrieval first, then re-check faithfulness."
        )
    if relevancy_bad:
        problems.append(
            "RELEVANCY: the answer doesn't closely address the question (verbose, "
            "off-topic, or over-hedged). Check whether the system prompt's refusal "
            "wording is firing on an answerable question."
        )
    if not problems:
        problems.append("OK — all scores above threshold.")

    return " | ".join(problems)


# STEP 4 — reporting

def write_markdown_report(path: str, label: str, per_row: List[dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    averages = {
        m: sum(r.get(m, 0) or 0 for r in per_row) / len(per_row)
        for m in metric_names
    }

    lines = []
    lines.append(f"# RAGAS Evaluation Report — `{label}`")
    lines.append("")
    lines.append(f"- **Timestamp:** {datetime.now().isoformat()}")
    lines.append(f"- **Test set size:** {len(per_row)}")
    lines.append("")
    lines.append("## Average scores")
    lines.append("")
    lines.append("| Metric | Average | Threshold |")
    lines.append("|--------|---------|-----------|")
    thresholds = {
        "faithfulness": FAITHFULNESS_THRESHOLD,
        "answer_relevancy": RELEVANCY_THRESHOLD,
        "context_precision": CONTEXT_PRECISION_THRESHOLD,
        "context_recall": CONTEXT_RECALL_THRESHOLD,
    }
    for m in metric_names:
        lines.append(f"| {m} | {averages[m]:.3f} | {thresholds[m]} |")
    lines.append("")
    lines.append(f"**Summary line:** this system is **{averages['faithfulness']*100:.0f}% faithful** "
                  f"and **{averages['answer_relevancy']*100:.0f}% relevant** on this {len(per_row)}-question set.")
    lines.append("")
    lines.append("## Per-question results")
    lines.append("")
    lines.append("| ID | Doc | Faithfulness | Relevancy | Ctx Precision | Ctx Recall | Diagnosis |")
    lines.append("|----|-----|--------------|-----------|----------------|------------|-----------|")
    for r in per_row:
        lines.append(
            f"| {r.get('id')} | {r.get('source_doc')} | "
            f"{r.get('faithfulness', 0):.2f} | {r.get('answer_relevancy', 0):.2f} | "
            f"{r.get('context_precision', 0):.2f} | {r.get('context_recall', 0):.2f} | "
            f"{diagnose(r)} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Full detail per question")
    lines.append("")
    for r in per_row:
        lines.append(f"### {r.get('id')} — {r.get('source_doc')}")
        lines.append("")
        lines.append(f"**Question:** {r.get('user_input', '')}")
        lines.append("")
        lines.append(f"**Reference (ground truth):** {r.get('reference', '')}")
        lines.append("")
        lines.append(f"**Actual answer:** {r.get('response', '')}")
        lines.append("")
        lines.append(f"**Scores:** faithfulness={r.get('faithfulness', 0):.2f}, "
                      f"answer_relevancy={r.get('answer_relevancy', 0):.2f}, "
                      f"context_precision={r.get('context_precision', 0):.2f}, "
                      f"context_recall={r.get('context_recall', 0):.2f}")
        lines.append("")
        lines.append(f"**Diagnosis:** {diagnose(r)}")
        lines.append("")
        lines.append("---")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return averages


def append_history_csv(path: str, label: str, averages: dict, n: int):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "label", "n_questions",
                              "faithfulness", "answer_relevancy",
                              "context_precision", "context_recall"])
        writer.writerow([
            datetime.now().isoformat(), label, n,
            f"{averages['faithfulness']:.4f}",
            f"{averages['answer_relevancy']:.4f}",
            f"{averages['context_precision']:.4f}",
            f"{averages['context_recall']:.4f}",
        ])


# MAIN

def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation against the RAG /chat endpoint.")
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="Backend base URL")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Per-request timeout (s)")
    parser.add_argument("--out", default="debug/ragas_report.md", help="Output Markdown report path")
    parser.add_argument("--history", default="debug/ragas_history.csv", help="Running CSV of average scores")
    parser.add_argument("--label", default="run", help="Label for this run, e.g. 'before' or 'after'")
    args = parser.parse_args()

    print("=" * 70)
    print(f"RAGAS evaluation run — label: {args.label}")
    print(f"Target: {args.url}")
    print(f"Test set size: {len(QA_DATASET)}")
    print("=" * 70 + "\n")

    rows = build_eval_rows(args.url, args.timeout)

    n_errors = sum(1 for r in rows if r["error"])
    if n_errors:
        print(f"\n[!] {n_errors}/{len(rows)} questions errored out and will be excluded from scoring.\n")

    per_row, _ = run_ragas(rows)

    averages = write_markdown_report(args.out, args.label, per_row)
    append_history_csv(args.history, args.label, averages, len(per_row))

    print("\n" + "=" * 70)
    print("AVERAGE SCORES")
    print("=" * 70)
    for k, v in averages.items():
        flag = "  <-- LOW" if v < 0.7 else ""
        print(f"  {k:20s}: {v:.3f}{flag}")

    print(f"\nFull report:  {args.out}")
    print(f"History log:  {args.history}")


if __name__ == "__main__":
    main()
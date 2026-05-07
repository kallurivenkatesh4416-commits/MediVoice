"""Standalone validation harness for the v19 deterministic eval layer.

Executes the code cells of medivoice_gemma4_v19_lab_report.ipynb in a shared
namespace with CPU smoke mode forced, stubs optional dependencies that are not
available locally (textstat, whisper), skips the pip-install cell, and asserts
that the deterministic layer passes for every case in the eval corpus.

Not meant to be shipped to Kaggle - this is a local sanity check."""
from __future__ import annotations

import json
import pathlib
import sys
import types


def install_stubs() -> None:
    if "textstat" not in sys.modules:
        textstat_stub = types.ModuleType("textstat")
        textstat_stub.flesch_kincaid_grade = lambda text: 6.0  # type: ignore[attr-defined]
        textstat_stub.flesch_reading_ease = lambda text: 70.0  # type: ignore[attr-defined]
        sys.modules["textstat"] = textstat_stub

    if "whisper" not in sys.modules:
        whisper_stub = types.ModuleType("whisper")
        def _missing(*args, **kwargs):  # noqa: ANN001, D401
            raise RuntimeError("whisper stub - not installed locally")
        whisper_stub.load_model = _missing  # type: ignore[attr-defined]
        sys.modules["whisper"] = whisper_stub

    import torch  # noqa: I001
    torch.cuda.is_available = lambda: False  # type: ignore[assignment]
    try:
        torch.cuda.is_bf16_supported = lambda: False  # type: ignore[assignment]
    except Exception:
        pass


def run() -> int:
    install_stubs()

    notebook_path = pathlib.Path("medivoice_gemma4_v19_lab_report.ipynb")
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))

    namespace: dict = {"__name__": "__main__", "display": lambda *args, **kwargs: None}
    skipped_reasons: list[tuple[int, str]] = []
    executed = 0

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        stripped = source.strip()
        if not stripped:
            continue

        if stripped.startswith("%%capture") or "!pip install" in stripped:
            skipped_reasons.append((idx, "pip install cell"))
            continue

        if "CUDA-enabled PyTorch" in source:
            skipped_reasons.append((idx, "CUDA repair cell"))
            continue

        tolerate_partial = "import gradio as gr" in source

        try:
            exec(compile(source, f"<cell {idx}>", "exec"), namespace)
            executed += 1
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            if tolerate_partial:
                print(f"\n[WARN] cell {idx} (gradio UI) raised {type(exc).__name__}: {exc}")
                print("  Treating as local gradio version skew - localize_* helpers should already be defined.")
                skipped_reasons.append((idx, f"gradio partial exec: {type(exc).__name__}"))
                continue
            print(f"\n[FAIL] cell {idx} raised {type(exc).__name__}: {exc}")
            head = source.splitlines()[:3]
            print("  Cell head: " + " | ".join(head))
            return 1

    print(f"\nExecuted {executed} code cells ({len(skipped_reasons)} skipped).")
    for idx, reason in skipped_reasons:
        print(f"  skip cell {idx}: {reason}")

    failures: list[str] = []

    eval_corpus = namespace.get("eval_corpus")
    if eval_corpus is None:
        failures.append("eval_corpus not defined")
    else:
        case_ids = [c["case_id"] for c in eval_corpus]
        print(f"\nEval corpus has {len(case_ids)} cases:")
        for cid in case_ids:
            print(f"  - {cid}")
        required = [
            "cbc_ocr_digit_swap_flag_mismatch",
            "cmp_missing_decimal_safe_escalation",
            "ped_cbc_neutropenia",
            "ped_cmp_bun_coverage_gap",
        ]
        for cid in required:
            if cid not in case_ids:
                failures.append(f"eval_corpus missing adversarial case {cid}")

    task_eval_df = namespace.get("task_eval_df")
    if task_eval_df is None:
        failures.append("task_eval_df not defined")
    else:
        print(f"\ntask_eval_df rows: {len(task_eval_df)}")
        status_match = task_eval_df["status_match"].mean()
        classification_acc = task_eval_df["deterministic_classification_accuracy"].dropna().mean()
        escalation = task_eval_df["safety_escalation_correct"].mean()
        citation = task_eval_df["citation_grounding_rate"].dropna().mean()
        flag_mismatch = task_eval_df["flag_mismatch_accuracy"].dropna().mean()
        print(f"  status_match_rate      : {status_match}")
        print(f"  classification_acc     : {classification_acc}")
        print(f"  escalation_pass_rate   : {escalation}")
        print(f"  citation_grounding     : {citation}")
        print(f"  flag_mismatch_accuracy : {flag_mismatch}")
        if status_match != 1.0:
            failures.append(f"status_match_rate={status_match} (expected 1.0)")
        if escalation != 1.0:
            failures.append(f"safety_escalation_pass_rate={escalation} (expected 1.0)")
        if classification_acc != 1.0:
            failures.append(f"classification_acc={classification_acc} (expected 1.0)")
        if citation != 1.0:
            failures.append(f"citation_grounding={citation} (expected 1.0)")
        if flag_mismatch != 1.0:
            failures.append(f"flag_mismatch_accuracy={flag_mismatch} (expected 1.0)")

    baseline_df = namespace.get("baseline_df")
    if baseline_df is not None:
        print(f"\nbaseline_df rows: {len(baseline_df)} (baselines x cases)")
        print(baseline_df.groupby("baseline")["escalation_correct"].mean().to_string())
    else:
        failures.append("baseline_df not defined")

    multilingual_df = namespace.get("multilingual_validation_df")
    if multilingual_df is not None:
        print(f"\nmultilingual_validation_df rows: {len(multilingual_df)}")
        if "disclaimer_preserved" in multilingual_df.columns:
            disclaimer_pct = multilingual_df["disclaimer_preserved"].mean()
            print(f"  disclaimer_preserved rate : {disclaimer_pct}")
            if disclaimer_pct != 1.0:
                failures.append(f"multilingual disclaimer_preserved={disclaimer_pct}")
        else:
            failures.append("multilingual_validation_df missing disclaimer_preserved")

        if "action_text_preserved" in multilingual_df.columns:
            action_pct = multilingual_df["action_text_preserved"].mean()
            print(f"  action_text_preserved rate: {action_pct}")
            if action_pct != 1.0:
                failures.append(f"multilingual action_text_preserved={action_pct}")
        else:
            failures.append("multilingual_validation_df missing action_text_preserved")
    else:
        skipped_reasons.append((-1, "multilingual_validation_df not defined"))

    safe_failure_df = namespace.get("safe_failure_df")
    if safe_failure_df is not None:
        print(f"\nsafe_failure_df rows: {len(safe_failure_df)}")
        scored = safe_failure_df[safe_failure_df["pass"].notna()]
        if len(scored):
            rate = scored["pass"].mean()
            print(f"  safe_failure pass rate: {rate}")
            if rate != 1.0:
                failures.append(f"safe_failure_pass_rate={rate}")
    else:
        failures.append("safe_failure_df not defined")

    coverage_smoke_df = namespace.get("coverage_smoke_df")
    if coverage_smoke_df is not None and not coverage_smoke_df.empty:
        print(f"\ncoverage_smoke_df rows: {len(coverage_smoke_df)}")
        cov_status = coverage_smoke_df["status_match"].mean()
        cov_class = coverage_smoke_df["classification_accuracy"].dropna().mean()
        cov_esc = coverage_smoke_df["escalation_correct"].mean()
        print(f"  coverage status_match     : {cov_status}")
        print(f"  coverage classification   : {cov_class}")
        print(f"  coverage escalation       : {cov_esc}")
        if cov_status != 1.0:
            failures.append(f"coverage_smoke status_match={cov_status} (expected 1.0)")
        if cov_class != 1.0:
            failures.append(f"coverage_smoke classification={cov_class} (expected 1.0)")
        if cov_esc != 1.0:
            failures.append(f"coverage_smoke escalation={cov_esc} (expected 1.0)")
    else:
        print("\ncoverage_smoke_df not defined or empty (non-blocking)")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nAll deterministic validations passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run())

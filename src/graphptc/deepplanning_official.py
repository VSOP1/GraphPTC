from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from threading import Lock


def _domain_path(root: Path, domain: str) -> Path:
    return root / "benchmark" / "deepplanning" / domain


def convert_travel(args: argparse.Namespace) -> int:
    domain = _domain_path(args.root, "travelplanning")
    sys.path.insert(0, str(domain))
    from agent.prompts import get_format_convert_prompt
    from evaluation.convert_report import process_single_report
    from openai import OpenAI

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"missing conversion API key: {args.api_key_env}")
    client = OpenAI(api_key=api_key, base_url=args.base_url, max_retries=0, timeout=args.timeout)
    reports = sorted((args.result_dir / "reports").glob("id_*.txt"))
    output = args.result_dir / "converted_plans"
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for report in reports:
        results.append(
            process_single_report(
                report, output, args.model, client, get_format_convert_prompt(args.language),
                Lock(), max_retries=args.max_retries,
            )
        )
    summary = {
        "total": len(results), "success": sum(bool(item.get("success")) for item in results),
        "failed": sum(not bool(item.get("success")) for item in results), "results": results,
        "model": args.model, "base_url": args.base_url, "max_retries": args.max_retries,
    }
    (args.result_dir / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if summary["failed"] == 0 else 1


def evaluate_travel(args: argparse.Namespace) -> int:
    domain = _domain_path(args.root, "travelplanning")
    sys.path.insert(0, str(domain))
    from evaluation.eval_converted import evaluate_plans

    result = evaluate_plans(
        result_dir=args.result_dir,
        test_data_path=domain / "data" / f"travelplanning_query_{args.language}.json",
        database_dir=domain / "database" / f"database_{args.language}",
        workers=args.workers,
        verbose=False,
    )
    return 0 if int(result.get("failed", 0)) == 0 else 1


def evaluate_shopping(args: argparse.Namespace) -> int:
    domain = _domain_path(args.root, "shoppingplanning")
    sys.path.insert(0, str(domain))
    from evaluation.evaluation_pipeline import (
        evaluate_single_case,
        generate_case_report,
        generate_summary_report,
    )

    selected = set(args.case_id)
    case_dirs = sorted(
        path for path in args.database_dir.glob("case_*")
        if path.is_dir() and (not selected or path.name.removeprefix("case_") in selected)
    )
    results = []
    for case_dir in case_dirs:
        try:
            results.append(evaluate_single_case(case_dir))
        except Exception as exc:
            results.append({
                "case_name": case_dir.name, "success": False,
                "error": f"{type(exc).__name__}: {exc}", "score": 0.0,
                "case_score": 0.0, "is_completed": False,
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        if result.get("success"):
            generate_case_report(result, args.output_dir)
    generate_summary_report(results, args.output_dir)
    failures = sum(bool(result.get("error")) for result in results)
    return 0 if failures == 0 else 1


def aggregate(args: argparse.Namespace) -> int:
    deepplanning = args.root / "benchmark" / "deepplanning"
    sys.path.insert(0, str(deepplanning))
    sys.path.insert(0, str(deepplanning / "shoppingplanning"))
    from aggregate_results import load_travel_statistics
    from shoppingplanning.evaluation.score_statistics import calculate_model_statistics

    shopping = calculate_model_statistics(args.method, args.shopping_report_dir)
    travel = load_travel_statistics(deepplanning / "travelplanning", args.method, str(args.travel_results_dir))
    shopping_weighted = shopping["total"]["weighted_average_case_score"] if shopping else None
    travel_acc = travel["total"]["weighted_average_case_score"] if travel else None
    result = {
        "method": args.method,
        "domains": {"shopping": shopping, "travel": travel},
        "overall": {
            "avg_acc": ((shopping_weighted + travel_acc) / 2) if shopping_weighted is not None and travel_acc is not None else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if shopping is not None and travel is not None else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("convert-travel", "evaluate-travel"):
        item = sub.add_parser(name)
        item.add_argument("--root", type=Path, required=True)
        item.add_argument("--result-dir", type=Path, required=True)
        item.add_argument("--language", choices=("zh", "en"), required=True)
        if name == "convert-travel":
            item.add_argument("--model", required=True)
            item.add_argument("--base-url", required=True)
            item.add_argument("--api-key-env", required=True)
            item.add_argument("--timeout", type=float, default=300)
            item.add_argument("--max-retries", type=int, default=30)
        else:
            item.add_argument("--workers", type=int, default=1)
    shopping = sub.add_parser("evaluate-shopping")
    shopping.add_argument("--root", type=Path, required=True)
    shopping.add_argument("--database-dir", type=Path, required=True)
    shopping.add_argument("--output-dir", type=Path, required=True)
    shopping.add_argument("--case-id", action="append", default=[])
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--root", type=Path, required=True)
    aggregate_parser.add_argument("--method", required=True)
    aggregate_parser.add_argument("--travel-results-dir", type=Path, required=True)
    aggregate_parser.add_argument("--shopping-report-dir", type=Path, required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    functions = {
        "convert-travel": convert_travel,
        "evaluate-travel": evaluate_travel,
        "evaluate-shopping": evaluate_shopping,
        "aggregate": aggregate,
    }
    with contextlib.redirect_stdout(sys.stderr):
        return functions[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

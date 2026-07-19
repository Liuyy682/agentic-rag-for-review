import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from evaluation.io import read_metrics_csv
from evaluation.reports import write_compare_report
from agentic_rag import config


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two evaluation metric CSV files.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--output", default=str(Path(config.EVALUATION_REPORTS_DIR) / "compare_report.md"))
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--current-label", default="Current")
    args = parser.parse_args()

    write_compare_report(
        args.output,
        baseline=read_metrics_csv(args.baseline),
        current=read_metrics_csv(args.current),
        baseline_label=args.baseline_label,
        current_label=args.current_label,
    )
    print(args.output)


if __name__ == "__main__":
    main()

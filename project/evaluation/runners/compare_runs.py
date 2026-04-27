import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from evaluation.io import read_metrics_csv
from evaluation.reports import write_compare_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two evaluation metric CSV files.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--output", default=str(PROJECT_DIR / "evaluation" / "reports" / "compare_report.md"))
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


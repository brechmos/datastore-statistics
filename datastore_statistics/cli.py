"""Command-line interface for datastore statistics."""

import argparse
import datetime
import getpass
import logging
import shutil
import socket
import sys
from collections import OrderedDict
from typing import Any, Optional

import pandas as pd
import tqdm

from .report import generate_html_report
from .stats import add_stats, compute_outliers
from .utils import find_files

OUTPUT_FORMAT_CHOICES = ["screen", "json", "csv"]
STATISTIC_CHOICES = ["zscore", "iqr"]
DIRECTION_CHOICES = ["absolute", "lower", "higher"]


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Find outliers in medical image datasets."
    )
    parser.add_argument(
        "--data-directory",
        type=str,
        dest="data_directory",
        help="Data directory containing images",
        default="data/",
    )
    parser.add_argument(
        "--filename-regexp",
        type=str,
        dest="filename_regexp",
        help="Regexp pattern for filtering filenames",
        default=r".*\.(jpeg|jpg|png|dcm|dicom)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        dest="metric",
        nargs="+",
        help="Metric(s) to check (e.g. mean median n_slices com_x)",
        default=["mean"],
    )
    parser.add_argument(
        "--statistic",
        type=str,
        dest="statistic",
        help="Statistic for flagging outliers (zscore, iqr)",
        choices=STATISTIC_CHOICES,
        default="zscore",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        dest="output_format",
        help="Output format (screen, json, csv)",
        choices=OUTPUT_FORMAT_CHOICES,
        default="screen",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        dest="threshold",
        help="Threshold for outlier detection",
        default=None,
    )
    parser.add_argument(
        "--threshold-direction",
        type=str,
        dest="threshold_direction",
        help="Direction for threshold (absolute, lower, higher)",
        choices=DIRECTION_CHOICES,
        default="absolute",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        dest="mask_threshold",
        help="Ignore pixels below this value when computing stats",
        default=None,
    )
    parser.add_argument(
        "--mask-auto",
        dest="mask_auto",
        action="store_true",
        help="Automatically determine mask threshold using Otsu's method",
    )
    parser.add_argument(
        "--logfile", type=str, dest="logfile", help="Optional log file path", default=""
    )
    parser.add_argument(
        "--html-report",
        type=str,
        dest="html_report",
        help="Generate HTML report with thumbnails at this path",
        default=None,
    )
    parser.add_argument(
        "--group-by-sidecar",
        type=str,
        dest="group_by_sidecar",
        nargs="+",
        help=(
            "JSON path(s) in sidecar file to group by"
            " (e.g., SeriesInfo.MagneticFieldStrength)"
        ),
        default=None,
    )
    return parser.parse_args(args)


def setup_logging(args: argparse.Namespace) -> logging.Logger:
    """Configure logging based on arguments."""
    log = logging.getLogger("datastore_statistics")
    log.setLevel(logging.INFO)

    formatter = logging.Formatter("%(message)s")

    # File handler
    if args.logfile:
        fh = logging.FileHandler(args.logfile, mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        log.addHandler(fh)

        log.info(
            f"File written on "
            f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} '
            f"by {getpass.getuser()} on {socket.gethostname()}"
        )
        log.info(" ")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.addHandler(ch)

    return log


def output_results(log: logging.Logger, df: "pd.DataFrame", output_format: str) -> None:
    """Output results in the specified format."""
    if output_format == "screen":
        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.max_colwidth",
            54,
            "display.width",
            shutil.get_terminal_size().columns,
        ):
            log.info(df)
    elif output_format == "csv":
        log.info(df.to_csv())
    elif output_format == "json":
        log.info(df.to_json())


def main(args: Optional[list[str]] = None) -> int:
    """Main entry point for the CLI."""
    parsed = parse_args(args)
    log = setup_logging(parsed)

    # Find files
    filenames = find_files(parsed.data_directory, parsed.filename_regexp)
    file_list: list[dict[str, Any]] = [OrderedDict({"filename": f}) for f in filenames]

    if not file_list:
        log.warning(
            f"No files found in {parsed.data_directory} "
            f"matching {parsed.filename_regexp}"
        )
        return 1

    # Compute stats
    failed: list[dict[str, Any]] = []
    for file in tqdm.tqdm(file_list, desc="Loading data...", file=sys.stdout):
        try:
            add_stats(
                file,
                mask_threshold=parsed.mask_threshold,
                mask_auto=parsed.mask_auto,
                metrics=set(parsed.metric),
            )
        except Exception as exc:
            log.warning(f"Skipping {file['filename']}: {exc}")
            failed.append(file)

    for f in failed:
        file_list.remove(f)

    if not file_list:
        log.warning("No files could be processed successfully")
        return 1

    # Group by sidecar field(s) if requested
    if parsed.group_by_sidecar:
        from .sidecar import get_group_value

        for file in file_list:
            parts = [
                get_group_value(file["filename"], jp) for jp in parsed.group_by_sidecar
            ]
            file["group"] = " | ".join(parts)

        groups: dict[str, list[dict[str, Any]]] = {}
        for file in file_list:
            groups.setdefault(file["group"], []).append(file)

        for gname, gfiles in groups.items():
            if len(gfiles) < 3:
                log.warning(
                    f"Group '{gname}' has only {len(gfiles)} file(s); "
                    f"z-scores may be unreliable"
                )
    else:
        groups = {"_all": file_list}

    # Compute outlier scores and apply threshold within each group
    from .utils import threshold as apply_threshold

    score_prefix = "zscore-" if parsed.statistic == "zscore" else "iqr_scale-"
    result_list: list[dict[str, Any]] = []
    all_scored: list[dict[str, Any]] = []
    group_totals: dict[str, int] = {}

    for gname, gfiles in sorted(groups.items()):
        group_totals[gname] = len(gfiles)
        for metric in parsed.metric:
            compute_outliers(
                gfiles,
                metric,
                parsed.statistic,
                threshold_value=None,
                threshold_direction=parsed.threshold_direction,
            )

        all_scored.extend(gfiles)

        if parsed.threshold is not None:
            gfiles = [
                f
                for f in gfiles
                if any(
                    apply_threshold(
                        float(f.get(score_prefix + m, float("nan"))),
                        parsed.threshold,
                        parsed.threshold_direction,
                    )
                    for m in parsed.metric
                )
            ]

        result_list.extend(gfiles)

    file_list = result_list

    # Output results
    df = pd.DataFrame.from_dict(file_list)
    # Convert Path objects to strings for JSON serialization
    if "filename" in df.columns:
        df["filename"] = df["filename"].astype(str)
    # Drop internal group column if not grouping
    if not parsed.group_by_sidecar and "group" in df.columns:
        df = df.drop(columns=["group"])
    output_results(log, df, parsed.output_format)

    # Generate HTML report if requested
    if parsed.html_report:
        report_path = generate_html_report(
            file_list,
            parsed.html_report,
            metrics=parsed.metric,
            threshold=parsed.threshold,
            group_key="group" if parsed.group_by_sidecar else None,
            group_totals=group_totals if parsed.group_by_sidecar else None,
            all_files=all_scored if parsed.threshold else None,
        )
        log.info(f"HTML report written to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

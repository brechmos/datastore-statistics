"""Integration tests for datastore_statistics CLI."""

import json
from pathlib import Path

import pytest

from datastore_statistics.cli import main, parse_args


class TestParseArgs:
    """Tests for argument parsing."""

    def test_default_args(self) -> None:
        """Test default argument values."""
        args = parse_args([])
        assert args.data_directory == "data/"
        assert args.metric == ["mean"]
        assert args.statistic == "zscore"
        assert args.output_format == "screen"
        assert args.threshold is None
        assert args.threshold_direction == "absolute"
        assert args.mask_threshold is None
        assert args.mask_auto is False


class TestMainIntegration:
    """Integration tests for main CLI function."""

    def test_main_with_test_data(
        self, test_data_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test main function runs successfully with test data."""
        result = main(
            [
                "--data-directory",
                str(test_data_dir),
                "--filename-regexp",
                r".*\.jpeg",
            ]
        )
        assert result == 0

    def test_main_no_files_found(
        self, temp_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test main returns 1 when no files found."""
        result = main(
            [
                "--data-directory",
                str(temp_dir),
                "--filename-regexp",
                r".*\.jpeg",
            ]
        )
        assert result == 1

    def test_main_with_threshold(self, test_data_dir: Path) -> None:
        """Test main with threshold filtering."""
        result = main(
            [
                "--data-directory",
                str(test_data_dir),
                "--filename-regexp",
                r".*\.jpeg",
                "--threshold",
                "2.0",
            ]
        )
        assert result == 0

    def test_main_csv_output(self, test_data_dir: Path, temp_dir: Path) -> None:
        """Test main with CSV output format."""
        log_path = temp_dir / "csv_output.log"
        result = main(
            [
                "--data-directory",
                str(test_data_dir),
                "--filename-regexp",
                r".*\.jpeg",
                "--output-format",
                "csv",
                "--logfile",
                str(log_path),
            ]
        )
        assert result == 0
        content = log_path.read_text()
        # CSV output should contain commas and headers
        assert "," in content
        assert "filename" in content

    def test_main_json_output(self, test_data_dir: Path, temp_dir: Path) -> None:
        """Test main with JSON output format."""
        log_path = temp_dir / "json_output.log"
        result = main(
            [
                "--data-directory",
                str(test_data_dir),
                "--filename-regexp",
                r".*\.jpeg",
                "--output-format",
                "json",
                "--logfile",
                str(log_path),
            ]
        )
        assert result == 0
        content = log_path.read_text()
        # Extract JSON part (skip header lines)
        lines = content.strip().split("\n")
        # Find line that starts with { (JSON output)
        json_line = next((line for line in lines if line.strip().startswith("{")), None)
        assert json_line is not None
        data = json.loads(json_line)
        assert "filename" in data or "mean" in data

    def test_main_with_mask_auto(self, test_data_dir: Path) -> None:
        """Test main with automatic masking."""
        result = main(
            [
                "--data-directory",
                str(test_data_dir),
                "--filename-regexp",
                r".*\.jpeg",
                "--mask-auto",
            ]
        )
        assert result == 0

    def test_main_with_iqr_statistic(self, test_data_dir: Path) -> None:
        """Test main with IQR statistic."""
        result = main(
            [
                "--data-directory",
                str(test_data_dir),
                "--filename-regexp",
                r".*\.jpeg",
                "--statistic",
                "iqr",
            ]
        )
        assert result == 0

    def test_main_html_report(self, test_data_dir: Path, temp_dir: Path) -> None:
        """Test main generates HTML report."""
        report_path = temp_dir / "report.html"
        result = main(
            [
                "--data-directory",
                str(test_data_dir),
                "--filename-regexp",
                r".*\.jpeg",
                "--html-report",
                str(report_path),
            ]
        )
        assert result == 0
        assert report_path.exists()
        content = report_path.read_text()
        assert "<html>" in content
        assert "Datastore Statistics Report" in content

    def test_main_logfile(self, test_data_dir: Path, temp_dir: Path) -> None:
        """Test main writes to logfile."""
        log_path = temp_dir / "output.log"
        result = main(
            [
                "--data-directory",
                str(test_data_dir),
                "--filename-regexp",
                r".*\.jpeg",
                "--logfile",
                str(log_path),
            ]
        )
        assert result == 0
        assert log_path.exists()
        content = log_path.read_text()
        assert "filename" in content.lower() or "mean" in content.lower()

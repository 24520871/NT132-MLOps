import json

import pandas as pd

from mlops.data import check
from tests.conftest import make_raw_frame


def _write_dataset(tmp_path, monkeypatch, frame: pd.DataFrame) -> None:
    path = tmp_path / "churn.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setenv("DATA_PATH", str(path))


def test_valid_dataset_passes(tmp_path, monkeypatch):
    _write_dataset(tmp_path, monkeypatch, make_raw_frame())

    result = check.validate_dataset("churn")

    assert result.passed
    assert result.dataset == "churn.csv"
    assert result.rows == 400
    assert result.columns == 21
    assert not result.failures


def test_missing_dataset_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "missing.csv"))

    result = check.validate_dataset("churn")

    assert not result.passed
    assert "was not found" in result.failures[0]


def test_empty_dataset_fails(tmp_path, monkeypatch):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    monkeypatch.setenv("DATA_PATH", str(path))

    result = check.validate_dataset("churn")

    assert not result.passed
    assert result.failures


def test_missing_feature_fails(tmp_path, monkeypatch):
    frame = make_raw_frame().drop(columns=["Contract"])
    _write_dataset(tmp_path, monkeypatch, frame)

    result = check.validate_dataset("churn")

    assert not result.passed
    assert "Contract" in result.failures[0]


def test_missing_target_fails(tmp_path, monkeypatch):
    frame = make_raw_frame().drop(columns=["Churn"])
    _write_dataset(tmp_path, monkeypatch, frame)

    result = check.validate_dataset("churn")

    assert not result.passed
    assert "Churn" in result.failures[0]


def test_invalid_target_labels_fail(tmp_path, monkeypatch):
    frame = make_raw_frame()
    frame.loc[0, "Churn"] = "Maybe"
    _write_dataset(tmp_path, monkeypatch, frame)

    result = check.validate_dataset("churn")

    assert not result.passed
    assert "Yes/No" in result.failures[0]


def test_target_missing_one_class_fails(tmp_path, monkeypatch):
    frame = make_raw_frame()
    frame["Churn"] = "No"
    _write_dataset(tmp_path, monkeypatch, frame)

    result = check.validate_dataset("churn")

    assert not result.passed
    assert "both classes" in result.failures[0]


def test_invalid_feature_type_fails(tmp_path, monkeypatch):
    frame = make_raw_frame()
    frame["tenure"] = frame["tenure"].astype(object)
    frame.loc[3, "tenure"] = "unknown"
    _write_dataset(tmp_path, monkeypatch, frame)

    result = check.validate_dataset("churn")

    assert not result.passed
    assert "tenure" in result.failures[0]


def test_invalid_nonblank_total_charges_fail(tmp_path, monkeypatch):
    frame = make_raw_frame()
    frame.loc[3, "TotalCharges"] = "not-a-number"
    _write_dataset(tmp_path, monkeypatch, frame)

    result = check.validate_dataset("churn")

    assert not result.passed
    assert "TotalCharges" in result.failures[0]


def test_missing_values_follow_existing_cleaning_policy(tmp_path, monkeypatch):
    frame = make_raw_frame()
    frame.loc[3, "MonthlyCharges"] = None
    _write_dataset(tmp_path, monkeypatch, frame)

    result = check.validate_dataset("churn")

    assert not result.passed
    assert "missing values" in result.failures[0]


def test_duplicate_csv_headers_fail(tmp_path, monkeypatch):
    path = tmp_path / "duplicate.csv"
    path.write_text("customerID,gender,gender\n1,Female,Male\n", encoding="utf-8")
    monkeypatch.setenv("DATA_PATH", str(path))

    result = check.validate_dataset("churn")

    assert not result.passed
    assert "duplicate CSV column names" in result.failures[0]


def test_cli_returns_success_exit_code_and_structured_output(tmp_path, monkeypatch, capsys):
    _write_dataset(tmp_path, monkeypatch, make_raw_frame())

    exit_code = check.main(["churn"])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["passed"] is True
    assert output["rows"] == 400


def test_cli_returns_failure_exit_code(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "missing.csv"))

    exit_code = check.main(["churn"])

    output = json.loads(capsys.readouterr().out)
    assert exit_code != 0
    assert output["passed"] is False


def test_unknown_usecase_fails_clearly(capsys):
    result = check.validate_dataset("unknown")

    assert not result.passed
    assert "only for use case 'churn'" in result.failures[0]

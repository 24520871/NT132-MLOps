import pytest

from mlops.config import load_config
from mlops.data import ingest, validation
from tests.conftest import make_raw_frame
from usecases.churn import data
from usecases.churn.schema import FEATURE_COLUMNS, SCHEMA, TARGET


def test_clean_produces_valid_frame():
    df = data.clean(make_raw_frame())
    validation.validate(df, SCHEMA)
    assert list(df.columns) == FEATURE_COLUMNS + [TARGET]
    assert "customerID" not in df.columns
    assert set(df[TARGET].unique()) <= {0, 1}


def test_blank_total_charges_become_zero():
    df = data.clean(make_raw_frame())
    assert df.loc[0, "TotalCharges"] == 0.0
    assert df["TotalCharges"].notna().all()


def test_missing_column_rejected():
    raw = make_raw_frame().drop(columns=["Contract"])
    with pytest.raises(validation.DataValidationError, match="Contract"):
        data.clean(raw)


def test_unexpected_category_rejected():
    df = data.clean(make_raw_frame())
    df.loc[1, "Contract"] = "Three year"
    with pytest.raises(validation.DataValidationError, match="Three year"):
        validation.validate(df, SCHEMA)


def test_negative_charge_rejected():
    df = data.clean(make_raw_frame())
    df.loc[2, "MonthlyCharges"] = -5.0
    with pytest.raises(validation.DataValidationError, match="MonthlyCharges"):
        validation.validate(df, SCHEMA)


def test_nulls_rejected():
    df = data.clean(make_raw_frame())
    df.loc[3, "tenure"] = None
    with pytest.raises(validation.DataValidationError, match="missing values"):
        validation.validate(df, SCHEMA)


def test_too_few_rows_rejected():
    df = data.clean(make_raw_frame()).head(10)
    with pytest.raises(validation.DataValidationError, match="rows"):
        validation.validate(df, SCHEMA)


def test_bad_target_rejected():
    raw = make_raw_frame()
    raw.loc[0, TARGET] = "Maybe"
    with pytest.raises(validation.DataValidationError, match="Yes/No"):
        data.clean(raw)


def test_missing_dataset_gives_clear_error(tmp_path):
    cfg = load_config("churn")
    cfg["data"]["path"] = str(tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError, match="never falls back"):
        data.load_data(cfg)


def test_split_is_stratified_and_deterministic():
    cfg = load_config("churn")
    df = data.clean(make_raw_frame(n=500))
    a = data.split_data(df, cfg)
    b = data.split_data(df, cfg)
    assert a[0].index.equals(b[0].index)
    assert abs(a[2].mean() - a[3].mean()) < 0.05
    assert len(a[1]) == pytest.approx(0.2 * 500, abs=1)
    assert ingest.file_sha256  # helper importable

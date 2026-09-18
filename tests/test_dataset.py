import pytest
import polars as pl

from src.data.dataset import CodeBEPADataSet

@pytest.fixture
def parquet_file(tmp_path):
    path = tmp_path / "test_dataset.parquet"

    data = pl.DataFrame({
        "problem": [
            "problem 1",
            "problem 2",
            "problem 3",
        ],
        "solution": [
            "solution 1",
            "solution 2",
            "solution 3",
        ],
    })

    data.write_parquet(path)

    return path


def test_dataset_accepts_path(parquet_file):
    dataset = CodeBEPADataSet(parquet_file)

    assert dataset._path == parquet_file


def test_dataset_accepts_string_path(parquet_file):
    dataset = CodeBEPADataSet(str(parquet_file))

    assert dataset._path == parquet_file


def test_dataset_raises_for_missing_file(tmp_path):
    missing_file = tmp_path / "does_not_exist.parquet"

    with pytest.raises(FileNotFoundError):
        CodeBEPADataSet(missing_file)


def test_dataset_raises_for_invalid_type():
    with pytest.raises(TypeError):
        CodeBEPADataSet(123)


def test_dataset_returns_all_pairs(parquet_file):
    dataset = CodeBEPADataSet(parquet_file)

    examples = list(dataset)

    assert examples == [
        ("problem 1", "solution 1"),
        ("problem 2", "solution 2"),
        ("problem 3", "solution 3"),
    ]


def test_dataset_returns_correct_number_of_examples(parquet_file):
    dataset = CodeBEPADataSet(parquet_file)

    examples = list(dataset)

    assert len(examples) == 3
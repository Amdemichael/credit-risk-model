import pytest
import pandas as pd
from src.data_processing import DataLoader

def test_data_loader_columns():
    """Test that DataLoader returns expected columns"""
    loader = DataLoader(file_path="tests/test_data.csv")
    df = loader.transform()
    expected_cols = {'customerid', 'transactionstarttime', 'value'}
    assert expected_cols.issubset(set(df.columns))

def test_data_loader_datetime():
    """Test datetime conversion"""
    loader = DataLoader(file_path="tests/test_data.csv")
    df = loader.transform()
    assert pd.api.types.is_datetime64_any_dtype(df['transactionstarttime'])
import pytest
import pandas as pd

from prj_tplt_dh import FillData

@pytest.fixture
def data_test():
    return pd.DataFrame({'col1': [1, 2], 'col2': [3, None], 'col3': ['A', None]})

def test_all_fill(data_test):
    filled = FillData(data_test)
    data_test = filled.fill_all()

    assert data_test.equals(pd.DataFrame({'col3': ['A', 'Miss'], 'col1': [1, 2], 'col2': [3.0, -9999]}))

def test_num_fill(data_test):
    filled = FillData(data_test)
    data_test = filled.fill_num()

    assert data_test.equals(pd.DataFrame({'col1': [1, 2], 'col2': [3.0, -9999]}))

def test_cat_fill(data_test):
    filled = FillData(data_test)
    data_test = filled.fill_cat()

    assert data_test.equals(pd.DataFrame({'col3': ['A', 'Miss']}))
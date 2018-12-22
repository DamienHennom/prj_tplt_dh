import pytest
import pandas as pd

from prj_tplt_dh import AverageResponse


@pytest.fixture
def data_test():
    return pd.DataFrame({'col1': [1, 2], 'col2': [3, 4], 'col3': ['A', 'B']})


def test_num_average_response(data_test):
    avg = AverageResponse(data_test, 'col1', bins=10)
    avg_data = avg.compute_average_response('col2')

    assert avg_data.equals(pd.DataFrame({'col2': ['3', '4'], 'col1': [1.0, 2.0], 'count': [1.0, 1.0]}))


def test_qual_average_response(data_test):
    avg = AverageResponse(data_test, 'col1', bins=10)
    avg_data = avg.compute_average_response('col3')

    assert avg_data.equals(pd.DataFrame({'col3': ['A', 'B'], 'col1': [1.0, 2.0], 'count': [1.0, 1.0]}))

import pytest
import numpy as np

from hashit.core import NumpyTable


def test_append():
    tbl = NumpyTable('test')
    tbl.clear_table()
    expected_data = []
    for i in range(100):
        data = np.random.rand(1,100)
        expected_data.append(data)
        tbl.append(data)

    expected = np.vstack(expected_data)
    print(expected.shape)
    for i in range(100):
        start, stop = np.random.randint(0,100,2)
        if start >= stop: continue
        np.testing.assert_array_equal(expected[start:stop], tbl[start:stop])

    for i in range(100):
        idx = int(np.random.randint(0,100,1))
        np.testing.assert_array_equal(expected[idx].reshape(1,-1), tbl[idx])





import pytest
import numpy as np

from diskhash.core import NumpyTable

test_data = [(np.dtype('float64')), (np.dtype('float32')), (np.dtype('int32')), (np.dtype('int64'))]
ids = ['dtype={0}'.format(str(dtype)) for dtype in test_data]
@pytest.mark.parametrize("dtype", test_data, ids=ids)
def test_append(dtype):
    tbl = NumpyTable('test')
    tbl.clear_table()
    expected_data = []
    for i in range(100):
        if dtype == np.dtype('int32') or dtype == np.dtype('int64'):
            data = np.array(np.random.randint(0,1000000, (1,100)), dtype=dtype)
        else:
            data = np.array(np.random.rand(1,100), dtype=dtype)
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


test_data = [(np.dtype('float64')), (np.dtype('float32')), (np.dtype('int32')), (np.dtype('int64'))]
ids = ['dtype={0}'.format(str(dtype)) for dtype in test_data]
@pytest.mark.parametrize("dtype", test_data, ids=ids)
def test_append_variable_length(dtype):
    tbl = NumpyTable('test', fixed_length=False)
    tbl.clear_table()
    expected_data = []
    for i in range(100):
        l = np.random.randint(1,100)
        if dtype == np.dtype('int32') or dtype == np.dtype('int64'):
            data = np.array(np.random.randint(0,1000000, (1,l)), dtype=dtype)
        else:
            data = np.array(np.random.rand(1,l), dtype=dtype)
        expected_data.append(data)
        tbl.append(data)

    for i in range(100):
        start, stop = np.random.randint(0,100,2)
        if start >= stop: continue
        arr = tbl[start:stop]
        expected_arr = expected_data[start:stop]

        max_l = 0
        for x in expected_arr:
            max_l = max(max_l, x.shape[1])

        assert arr.shape[1] == max_l, 'Array has the wrong maximum length. {0} but {1} was expected.'.format(arr.shape[1], max_l)
        assert arr.shape[0] == len(expected_arr), 'Number of samples not equal: {0} but {1} was expected.'.format(arr.shape[0], len(expected))
        for idx in range(stop-start):
            x = expected_arr[idx]
            np.testing.assert_array_equal(arr[idx][:x.shape[1]], x.reshape(-1))
            np.testing.assert_array_equal(arr[idx][x.shape[1]:], np.zeros((max_l - x.shape[1])))


test_data = [(np.dtype('float64')), (np.dtype('float32')), (np.dtype('int32')), (np.dtype('int64'))]
ids = ['dtype={0}'.format(str(dtype)) for dtype in test_data]
@pytest.mark.parametrize("dtype", test_data, ids=ids)
def test_append_variable_length_no_padding(dtype):
    tbl = NumpyTable('test', fixed_length=False, pad=False)
    tbl.clear_table()
    expected_data = []
    for i in range(100):
        l = np.random.randint(1,100)
        if dtype == np.dtype('int32') or dtype == np.dtype('int64'):
            data = np.array(np.random.randint(0,1000000, (1,l)), dtype=dtype)
        else:
            data = np.array(np.random.rand(1,l), dtype=dtype)
        expected_data.append(data)
        tbl.append(data)

    for i in range(100):
        start, stop = np.random.randint(0,100,2)
        if start >= stop: continue
        arr = tbl[start:stop]
        expected_arr = expected_data[start:stop]

        assert len(arr) == len(expected_arr), 'Number of samples not equal: {0} but {1} was expected.'.format(arr.shape[0], len(expected))
        for idx in range(stop-start):
            x = expected_arr[idx]
            x2 = arr[idx]
            np.testing.assert_array_equal(x2, x.reshape(-1))


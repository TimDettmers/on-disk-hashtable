import pytest
import numpy as np

from diskhash.core import NumpyTable
from diskhash.utils import Timer

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


test_data = [(np.dtype('float64')), (np.dtype('float32')), (np.dtype('int32')), (np.dtype('int64'))]
ids = ['dtype={0}'.format(str(dtype)) for dtype in test_data]
@pytest.mark.parametrize("dtype", test_data, ids=ids)
def test_numpy_and_list_indexing(dtype):
    index_length = 10
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
    for i in range(100):
        idx = np.random.randint(0,100,10)
        np.testing.assert_array_equal(expected[idx], tbl[idx])
        np.testing.assert_array_equal(expected[idx], tbl[idx.tolist()])


test_data = [(np.dtype('float64')), (np.dtype('float32')), (np.dtype('int32')), (np.dtype('int64'))]
ids = ['dtype={0}'.format(str(dtype)) for dtype in test_data]
@pytest.mark.parametrize("dtype", test_data, ids=ids)
def test_variable_length_np_indexing(dtype):
    tbl = NumpyTable('test', fixed_length=False)
    tbl.clear_table()
    expected_data = []
    index_length = 10
    for i in range(100):
        l = np.random.randint(1,100)
        if dtype == np.dtype('int32') or dtype == np.dtype('int64'):
            data = np.array(np.random.randint(0,1000000, (1,l)), dtype=dtype)
        else:
            data = np.array(np.random.rand(1,l), dtype=dtype)
        expected_data.append(data)
        tbl.append(data)

    for i in range(100):
        indices = np.random.randint(0,100,index_length)
        arr = tbl[indices]
        expected_arr = []
        for idx in indices:
            expected_arr.append(expected_data[idx])

        max_l = 0
        for x in expected_arr:
            max_l = max(max_l, x.shape[1])

        assert arr.shape[1] == max_l, 'Array has the wrong maximum length. {0} but {1} was expected.'.format(arr.shape[1], max_l)
        assert arr.shape[0] == len(expected_arr), 'Number of samples not equal: {0} but {1} was expected.'.format(arr.shape[0], len(expected))
        for idx in range(arr.shape[0]):
            x = expected_arr[idx]
            np.testing.assert_array_equal(arr[idx][:x.shape[1]], x.reshape(-1))
            np.testing.assert_array_equal(arr[idx][x.shape[1]:], np.zeros((max_l - x.shape[1])))


test_data = [(np.dtype('float64')), (np.dtype('float32')), (np.dtype('int32')), (np.dtype('int64'))]
ids = ['dtype={0}'.format(str(dtype)) for dtype in test_data]
@pytest.mark.parametrize("dtype", test_data, ids=ids)
def test_variable_length_no_padding_np_indexing(dtype):
    tbl = NumpyTable('test', fixed_length=False, pad=False)
    tbl.clear_table()
    expected_data = []
    index_length = 10
    for i in range(100):
        l = np.random.randint(1,100)
        if dtype == np.dtype('int32') or dtype == np.dtype('int64'):
            data = np.array(np.random.randint(0,1000000, (1,l)), dtype=dtype)
        else:
            data = np.array(np.random.rand(1,l), dtype=dtype)
        expected_data.append(data)
        tbl.append(data)

    for i in range(100):
        expected_arr = []
        indices = np.random.randint(0,100,10)
        arr = tbl[indices]
        for idx in indices:
            expected_arr.append(expected_data[idx])

        assert len(arr) == len(expected_arr), 'Number of samples not equal: {0} but {1} was expected.'.format(arr.shape[0], len(expected, arr))
        for idx in range(len(arr)):
            x = expected_arr[idx]
            x2 = arr[idx]
            np.testing.assert_array_equal(x2, x.reshape(-1))

test_data = [(np.dtype('float64')), (np.dtype('float32')), (np.dtype('int32')), (np.dtype('int64'))]
ids = ['dtype={0}'.format(str(dtype)) for dtype in test_data]
@pytest.mark.parametrize("dtype", test_data, ids=ids)
def test_index_select(dtype):
    tbl = NumpyTable('test', fixed_length=False, pad=False)
    tbl.clear_table()
    tbl.add_index('test', lambda x: x.shape[1])
    expected_data = {}
    expected_indices = {}
    for i in range(1,10):
        expected_data[i] = []
        expected_indices[i] = []

    for i in range(100):
        l = np.random.randint(1,10)
        if dtype == np.dtype('int32') or dtype == np.dtype('int64'):
            data = np.array(np.random.randint(0,1000000, (1,l)), dtype=dtype)
        else:
            data = np.array(np.random.rand(1,l), dtype=dtype)
        expected_data[l].append(data)
        expected_indices[l].append(i)

        tbl.append(data)

    for i in range(100):
        l = np.random.randint(1,10)
        limit = np.random.randint(2,100)
        indices = tbl.select_index('test', where=l, limit=limit)
        arr = tbl[indices]

        expected_arr = expected_data[l][:limit]

        np.testing.assert_array_equal(indices, expected_indices[l][:limit])
        assert len(arr) == len(expected_arr), 'Number of samples not equal: {0} but {1} was expected.'.format(arr.shape[0], len(expected_arr))
        for idx in range(len(arr)):
            x = expected_arr[idx]
            x2 = arr[idx]
            x3 = tbl[indices[idx]]
            np.testing.assert_array_equal(x, x3)
            np.testing.assert_array_equal(x2, x.reshape(-1))

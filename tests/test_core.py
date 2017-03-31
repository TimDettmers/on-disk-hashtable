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


test_data = [(np.dtype('float64')), (np.dtype('float32')), (np.dtype('int32')), (np.dtype('int64'))]
ids = ['dtype={0}'.format(str(dtype)) for dtype in test_data]
@pytest.mark.parametrize("dtype", test_data, ids=ids)
def test_get_random_batch(dtype):
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

        tbl.append(data)

    # use same seed as batcher
    rdm = np.random.RandomState(234325)
    batch_size = 12
    for i in range(100):
        idx = rdm.choice(100,batch_size, replace=False)
        batch1 = tbl[idx]
        batch2 = tbl.get_random_batch(batch_size)
        l = len(batch1)
        assert l == batch_size, 'Sample size of the fetched data not equal to the batch size!'
        for i in range(l):
            np.testing.assert_array_equal(batch1[i], batch2[i], 'Random batch not equal!')


test_data = [(np.dtype('float64')), (np.dtype('float32')), (np.dtype('int32')), (np.dtype('int64'))]
ids = ['dtype={0}'.format(str(dtype)) for dtype in test_data]
@pytest.mark.parametrize("dtype", test_data, ids=ids)
def test_get_index_random_batch(dtype):
    tbl = NumpyTable('test', fixed_length=False, pad=False)
    tbl.clear_table()
    tbl.add_index('test', lambda x: x.shape[1])
    expected_data = {}
    expected_indices = {}
    index_counts = {}
    for i in range(1,10):
        expected_data[i] = []
        expected_indices[i] = []
        index_counts[i] = 0

    for i in range(100):
        l = np.random.randint(1,10)
        if dtype == np.dtype('int32') or dtype == np.dtype('int64'):
            data = np.array(np.random.randint(0,1000000, (1,l)), dtype=dtype)
        else:
            data = np.array(np.random.rand(1,l), dtype=dtype)
        expected_data[l].append(data)
        expected_indices[l].append(i)
        index_counts[l] +=1

        tbl.append(data)

    # use same seed as batcher
    rdm = np.random.RandomState(234325)
    batch_size = 4
    p = np.array(index_counts.values(), dtype=np.float32)
    p = p/np.sum(p)
    for i in range(100):
        choice = rdm.choice(9,1, p=p)[0]
        idx = expected_data.keys()[choice]
        batch_idx = rdm.choice(expected_indices[idx], batch_size, replace=False)

        l = len(tbl[batch_idx])
        assert l == batch_size, 'Sample size of the fetched data not equal to the batch size!'
        batch1 = tbl[batch_idx]
        batch2 = tbl.get_random_index_batch(batch_size, 'test')
        for i in range(l):
            np.testing.assert_array_equal(batch1[i], batch2[i], 'Random batch not equal!')


test_data = [(np.dtype('float64')), (np.dtype('float32')), (np.dtype('int32')), (np.dtype('int64'))]
ids = ['dtype={0}'.format(str(dtype)) for dtype in test_data]
@pytest.mark.parametrize("dtype", test_data, ids=ids)
def test_join(dtype):
    tbl1 = NumpyTable('test', fixed_length=False, pad=False)
    tbl1.clear_table()
    tbl1.add_index('test', lambda x: x.shape[1])
    tbl2 = NumpyTable('test2', fixed_length=False, pad=False)
    tbl2.clear_table()
    tbl2.add_index('test', lambda x: x.shape[1])

    test_tbl1 = NumpyTable('test3', fixed_length=False, pad=False)
    test_tbl1.clear_table()
    test_tbl1.add_index('test', lambda x: x.shape[1])
    test_tbl2 = NumpyTable('test4', fixed_length=False, pad=False)
    test_tbl2.clear_table()
    test_tbl2.add_index('test', lambda x: x.shape[1])

    tbls = [tbl1, tbl2]
    test_tbls = [test_tbl1, test_tbl2]
    expected_data = [{}, {}]
    for i in range(1,10):
        expected_data[0][i] = []
        expected_data[1][i] = []

    for tbl_idx in range(2):
        for i in range(100):
            l = np.random.randint(1,10)
            if dtype == np.dtype('int32') or dtype == np.dtype('int64'):
                data = np.array(np.random.randint(0,1000000, (1,l)), dtype=dtype)
            else:
                data = np.array(np.random.rand(1,l), dtype=dtype)
            expected_data[tbl_idx][l].append(data)

            tbls[tbl_idx].append(data)
            test_tbls[tbl_idx].append(data)

    # use same seed as batcher
    tbl1.join(tbl2)
    batch_size = 4
    for i in range(100):
        idx = np.random.choice(100,batch_size, replace=False)
        l = len(tbl1[idx][0])
        assert l == batch_size, 'Sample size of the fetched data not equal to the batch size!'
        batch11 = tbl1[idx][0]
        batch12 = tbl1[idx][1]
        batch21 = test_tbls[0][idx]
        batch22 = test_tbls[1][idx]
        for i in range(l):
            np.testing.assert_array_equal(batch11[i], batch21[i])
            np.testing.assert_array_equal(batch12[i], batch22[i])



test_data = [(np.dtype('float64')), (np.dtype('float32')), (np.dtype('int32')), (np.dtype('int64'))]
ids = ['dtype={0}'.format(str(dtype)) for dtype in test_data]
@pytest.mark.parametrize("dtype", test_data, ids=ids)
def test_joined_index_batch(dtype):
    tbl1 = NumpyTable('test', fixed_length=False, pad=False)
    tbl1.clear_table()
    tbl1.add_index('test', lambda x: x.shape[1])
    tbl2 = NumpyTable('test2', fixed_length=False, pad=False)
    tbl2.clear_table()
    tbl2.add_index('test', lambda x: x.shape[1])
    tbls = [tbl1, tbl2]
    tbl1.join(tbl2)

    for tbl_idx in [0,1]:
        for i in range(100):
            l = np.random.randint(1,3)
            if dtype == np.dtype('int32') or dtype == np.dtype('int64'):
                data = np.array(np.random.randint(0,1000000, (1,l)), dtype=dtype)
            else:
                data = np.array(np.random.rand(1,l), dtype=dtype)

            tbls[tbl_idx].append(data)

    batch_size = 4
    for i in range(100):
        l = np.random.randint(1,3)
        batches = tbl1.get_random_index_batch(batch_size, 'test')
        assert len(batches) == 2, 'There should be two batches; one for each joined table.'
        assert len(batches[0]) == len(batches[1]), 'Batches should have the same length.'
        assert len(batches[0]) == batch_size, 'Batch length should be the batch size.'
        l1 = batches[0][0].shape[0]
        l2 = batches[1][0].shape[0]
        for i in range(len(batches[0])):
            assert batches[0][i].shape[0] == l1, 'All samples in a index batch should have the same length'
            assert batches[1][i].shape[0] == l2, 'All samples in a index batch should have the same length'

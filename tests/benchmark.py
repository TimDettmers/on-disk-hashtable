import numpy as np
from diskhash.core import NumpyTable
from diskhash.utils import Timer

rdm = np.random.RandomState(2345)
t = Timer()
write = False
numbers_per_vector = 32000

if write:
    tbl = NumpyTable('test')
    tbl.clear_table()
    expected_data = []
    for i in range(10000):
        data = np.random.rand(1,numbers_per_vector)
        expected_data.append(data)
        tbl.append(data)

    expected = np.vstack(expected_data)

else:
    tbl = NumpyTable('test')
    tbl.init()
    length = 2
    total_bytes = 0
    for i in range(100):
        start = rdm.randint(0,10000-length,)
        total_bytes += length*numbers_per_vector*8
        t.tick()
        a = tbl[start:start+length]
        t.tick()
    time = t.tock()

    MB = total_bytes/(1024**2.)
    print('Reading random access at {0} MB/s or {1} bytes in {2} seconds in chunks of {3} bytes'.format(MB/time, total_bytes, time, numbers_per_vector*length*8))



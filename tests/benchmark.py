import numpy as np
from diskhash.core import NumpyTable
from diskhash.utils import Timer


tbl = NumpyTable('test')
tbl.clear_table()
expected_data = []
numbers_per_vector = 2
for i in range(10000):
    data = np.random.rand(1,numbers_per_vector)
    expected_data.append(data)
    tbl.append(data)

expected = np.vstack(expected_data)

rdm = np.random.RandomState(2345)
t = Timer()

length = 1
total_bytes = 0
for i in range(500000):
    start = rdm.randint(0,10000-length,)
    total_bytes += length*128*8
    t.tick()
    tbl[start:start+length]
    t.tick()
time = t.tock()

MB = total_bytes/(1024**2.)
print('Reading random access at {0} MB/s or {1} bytes in {2} seconds in chunks of {3} bytes'.format(MB/time, total_bytes, time, expected.shape[1]*length*8))



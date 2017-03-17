import numpy as np
from hashit.core import NumpyTable
from hashit.utils import Timer


tbl = NumpyTable('test')
tbl.clear_table()
expected_data = []
for i in range(10000):
    data = np.random.rand(1,128)
    expected_data.append(data)
    tbl.append(data)

expected = np.vstack(expected_data)

rdm = np.random.RandomState(2345)
t = Timer()

length = 4
total_bytes = 0
for i in range(1000):
    start = rdm.randint(0,10000-length,)
    total_bytes += length*128*8
    t.tick()
    tbl[start:start+length]
    t.tick()
time = t.tock()

MB = total_bytes/(1024**2.)
print('Reading at {0} MB/s'.format(MB/time))


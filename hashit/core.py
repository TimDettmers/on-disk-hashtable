from __future__ import print_function
from os.path import join
from enum import Enum

import numpy as np
import redis
import mmap
import os
import sys

from hashit.utils import Timer
from numpy.testing import assert_array_equal

t = Timer()

class HashType(Enum):
    int32 = 0x1
    int64 = 0x2
    float32 = 0x3
    float64 = 0x4

np2HashType = {}
np2HashType[np.dtype('int32')] = HashType.int32
np2HashType[np.dtype('int64')] = HashType.int64
np2HashType[np.dtype('float32')] = HashType.float32
np2HashType[np.dtype('float64')] = HashType.float64

np2byte = {}
np2byte[np.dtype('int32')] = 4
np2byte[np.dtype('int64')] = 8
np2byte[np.dtype('float32')] = 4
np2byte[np.dtype('float64')] = 8

hashType2np = {}
hashType2np[HashType.int32.value] = np.dtype('int32')
hashType2np[HashType.int64.value] = np.dtype('int64')
hashType2np[HashType.float32.value] = np.dtype('float32')
hashType2np[HashType.float64.value] = np.dtype('float64')

def make_table_path(name):
    home = os.environ['HOME']
    base_dir = join(home, '.hashit')
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    if not os.path.exists(join(base_dir, name)):
        with open(join(base_dir, name), 'a'):
            os.utime(join(base_dir, name), None)
    return join(base_dir, name)


class NumpyTable(object):
    def __init__(self, name, fixed_length=True, pad=True):
        self.db = redis.StrictRedis(host='localhost', port=6379, db=0)
        self.name = name
        self.path = None
        self.fhandle = None
        self.length = 0
        self.fixed_length = fixed_length
        self.sync_with_redis()
        self.pad = pad

    def clear_table(self):
        if self.fhandle is not None:
            self.fhandle.close()
        if self.path is not None and os.path.exists(self.path):
            os.remove(self.path)
        self.db.delete(self.name + '.length')
        self.db.delete(self.name + '.idx_counter')
        self.db.delete(self.name + '.index')
        self.db.delete(self.name)
        self.sync_with_redis()

    def sync_with_redis(self):
        tbl_exists = True
        name = self.db.get(self.name)
        if name is None:
            tbl_exists = False

        if tbl_exists:
            self.path = self.db.get(self.name)
            self.fhandle = open(self.path, 'r+b')
            self.length = int(self.get('length'))
            self.idx = int(self.get('idx_counter'))
        else:
            self.path = make_table_path(self.name)
            self.fhandle = open(self.path, 'r+b')
            self.db.set(self.name, self.path)
            self.set('length', 0)
            self.set('idx_counter', 0)
            self.length = 0
            self.idx = 0

    def get(self, key):
        return self.db.get(self.name + '.' + key)

    def set(self, key, value):
        return self.db.set(self.name + '.' + key, value)

    def __del__(self):
        self.fhandle.close()

    def set_idx(self, idx, start, end, dtype, shape):
        if idx is None: idx = self.idx
        strvalue = str(start) + ' ' + str(end) + ' ' + str(dtype)
        for dim in shape:
            strvalue += ' ' + str(dim)
        key = self.name + '.index'
        self.db.hmset(key, {str(idx) : strvalue})
        self.idx +=1
        self.set('idx_counter', self.idx)

    def get_indices(self, indices):
        if isinstance(indices, list):
            stridx = [str(idx) for idx in indices]
        else:
            stridx = [str(indices)]

        strvalues = self.db.hmget(self.name + '.index', stridx)
        idx_values = []
        min_start = sys.maxint
        max_end = 0
        total_bytes = 0
        total_shape = None
        for idx, value in zip(stridx, strvalues):
            values = value.split(' ')
            start = int(values[0])
            end = int(values[1])
            dtype = hashType2np[int(values[2])]
            shape = [int(dim) for dim in values[3:]]
            min_start = min(start, min_start)
            max_end = max(end, max_end)
            total_bytes += end - start
            if total_shape is None:
                total_shape = shape
            else:
                total_shape[0] += shape[0]

            idx_values.append([idx, start, end, dtype, shape])
        return min_start, max_end, total_bytes, dtype, total_shape, idx_values

    def append(self, nparray, idx=None):
        self.fhandle.seek(self.length)
        start = self.fhandle.tell()
        typevalue = np2HashType[nparray.dtype].value
        bytearr = nparray.tobytes()
        self.fhandle.write(bytearr)
        self.length += len(bytearr)
        end = self.fhandle.tell()
        self.set_idx(self.idx, start, end, typevalue, nparray.shape)
        self.set('length', self.length)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            if step is not None:
                raise Exception('Step size not supported yet.')
            start, stop, total_bytes, dtype, shape, idx_values = self.get_indices(range(start, stop))
        elif isinstance(key, int):
            start, stop, total_bytes, dtype, shape, idx_values = self.get_indices(key)
        else:
            raise Exception('Unsupported sice type: {0}'.format(type(key)))

        assert stop-start == total_bytes, 'Non-contiguous access not supported yet'

        if self.fixed_length:
            self.fhandle.seek(start)
            data = np.frombuffer(self.fhandle.read(total_bytes), dtype=dtype)
            return data.reshape(shape)
        else:
            self.fhandle.seek(start)
            full_bytes = self.fhandle.read(total_bytes)
            byte = np2byte[dtype]
            if self.pad:
                max_length = 0
                global_start = start
                for idx, start, end, dtype, local_shape in idx_values:
                    max_length = max(max_length, (end-start)/byte)
                batch = np.empty((shape[0], max_length), dtype=dtype)
                for i, (idx, start, end, dtype, local_shape) in enumerate(idx_values):
                    start -= global_start
                    end -= global_start
                    batch[i, :(end-start)/byte] = np.frombuffer(full_bytes[start:end], dtype=dtype)
                    batch[i, (end-start)/byte:] = 0
                return batch
            else:
                batch = []
                max_length = 0
                global_start = start
                for i, (idx, start, end, dtype, local_shape) in enumerate(idx_values):
                    start -= global_start
                    end -= global_start
                    batch.append(np.frombuffer(full_bytes[start:end], dtype=dtype))
                return batch





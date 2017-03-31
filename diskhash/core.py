from __future__ import print_function
from os.path import join
from enum import Enum

import dill
import numpy as np
import mmap
import os
import sys

from diskhash.utils import Timer
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

def get_dill_data(name):
    home = os.environ['HOME']
    pkl_file = join(home, '.diskhash', name + '_index.pkl')
    if not os.path.exists(join(home, '.diskhash')):
        os.mkdir(join(home, '.diskhash'))
    if not os.path.exists(pkl_file):
        with open(pkl_file, 'a'):
            os.utime(pkl_file, None)
        return False, pkl_file, {}
    else:
        index = dill.load(open(pkl_file))

        return True, pkl_file, index

def make_table_path(name):
    home = os.environ['HOME']
    base_dir = join(home, '.diskhash')
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    if not os.path.exists(join(base_dir, name)):
        with open(join(base_dir, name), 'a'):
            os.utime(join(base_dir, name), None)
    return join(base_dir, name)


class NumpyTable(object):
    def __init__(self, name, fixed_length=True, pad=True):
        self.db = None
        self.name = name
        self.path = None
        self.index_path = join(os.environ['HOME'], '.diskhash', name + '_index.pkl')
        self.fhandle = None
        self.length = 0
        self.fixed_length = fixed_length
        self.pad = pad

    def init(self):
        self.sync_with_dill()

    def clear_table(self):
        if self.fhandle is not None:
            self.fhandle.close()
        if self.path is not None and os.path.exists(self.path):
            os.remove(self.path)
        tbl_path = make_table_path(self.name)
        if os.path.exists(tbl_path):
            os.remove(tbl_path)
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        self.sync_with_dill()

    def sync_with_dill(self):
        tbl_exists, path, index = get_dill_data(self.name)
        self.db = index

        if tbl_exists:
            self.path = self.db[self.name]
            self.fhandle = open(self.path, 'r+b')
            self.length = self.db[self.name + '.length']
            self.idx = self.db[self.name + '.idx_counter']
        else:
            self.path = make_table_path(self.name)
            self.fhandle = open(self.path, 'r+b')
            self.db[self.name] = self.path
            self.db[self.name + '.length'] = 0
            self.db[self.name + '.idx_counter'] = 0
            self.length = 0
            self.idx = 0
            self.db[self.name + '.indices'] = {}
            self.db[self.name + '.index_funcs'] = {}
            self.write_index()

    def add_index(self, index_name, index_func):
        assert index_name not in self.db[self.name + '.indices'], 'Index already present, cannot be overwritten!'
        self.db[self.name + '.indices'][index_name] =  {}
        self.db[self.name + '.index_funcs'][index_name] = dill.dumps(index_func)

    def select_index(self, index_name, where=None, limit=None):
        index = self.db[self.name + '.indices'][index_name]
        if where is not None:
            index = index[where]
        if limit is not None:
            index = index[:limit]

        return index


    def write_index(self):
        error = True
        while error:
            dill.dump(self.db, open(self.index_path + '.tmp', 'w'))
            try:
                idx = dill.load(open(self.index_path + '.tmp'))
            except:
                continue
            os.remove(self.index_path)
            os.rename(self.index_path + '.tmp', self.index_path)
            error = False

    def __del__(self):
        self.write_index()
        self.fhandle.close()

    def set_idx(self, idx, start, end, dtype, shape):
        if idx is None: idx = self.idx
        strvalue = str(start) + ' ' + str(end) + ' ' + str(dtype)
        for dim in shape:
            strvalue += ' ' + str(dim)
        key = self.name + '.index'
        if key not in self.db: self.db[key] = {}
        self.db[key][idx] = strvalue
        self.idx +=1
        self.db[self.name + '.idx_counter'] = self.idx

    def get_indices(self, indices):
        if isinstance(indices, list):
            stridx = [idx for idx in indices]
        else:
            stridx = [indices]

        strvalues = [self.db[self.name + '.index'][idx] for idx in stridx]
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
            if total_shape is None:
                total_shape = shape
            else:
                total_shape[0] += shape[0]

            idx_values.append([idx, start, end, dtype, shape])
        return min_start, max_end, total_bytes, dtype, total_shape, idx_values

    def append(self, nparray):
        for index_name in self.db[self.name + '.indices']:
            index_dict = self.db[self.name + '.indices'][index_name]
            strfunc = self.db[self.name + '.index_funcs'][index_name]
            func = dill.loads(strfunc)
            value = func(nparray)
            if value not in index_dict: index_dict[value] = []
            index_dict[value].append(self.idx)

        self.fhandle.seek(self.length)
        start = self.fhandle.tell()
        typevalue = np2HashType[nparray.dtype].value
        bytearr = nparray.tobytes()
        self.fhandle.write(bytearr)
        self.length += len(bytearr)
        end = self.fhandle.tell()
        self.set_idx(self.idx, start, end, typevalue, nparray.shape)
        self.db[self.name + '.length'] = self.length

    def padded_load(self, loaded_data, idx_values, samples, global_start):
        max_length = 0
        for idx, start, end, dtype, shape in idx_values:
            max_length = max(max_length, (end-start))
        byte = np2byte[dtype]
        max_length /= byte
        batch = np.empty((samples, max_length), dtype=dtype)
        for i, (idx, start, end, dtype, shape) in enumerate(idx_values):
            start -= global_start
            end -= global_start
            batch[i, :(end-start)/byte] = np.frombuffer(loaded_data[start:end], dtype=dtype)
            batch[i, (end-start)/byte:] = 0
        return batch

    def unpadded_load(self, loaded_data, idx_values, global_start):
        batch = []
        for i, (idx, start, end, dtype, local_shape) in enumerate(idx_values):
            start -= global_start
            end -= global_start
            batch.append(np.frombuffer(loaded_data[start:end], dtype=dtype))
        return batch

    def variable_length_noncontiguous_load(self, idx_values, samples):
        if self.pad:
            max_length = 0
            for idx, start, end, dtype, shape in idx_values:
                max_length = max(max_length, (end-start))
            byte = np2byte[dtype]
            max_length /= byte
            batch = np.empty((samples, max_length), dtype=dtype)
            for i, (idx, start, end, dtype, shape) in enumerate(idx_values):
                self.fhandle.seek(start)
                data = self.fhandle.read(end-start)
                batch[i, :(end-start)/byte] = np.frombuffer(data, dtype=dtype)
                batch[i, (end-start)/byte:] = 0
            return batch
        else:
            batch = []
            for i, (idx, start, end, dtype, local_shape) in enumerate(idx_values):
                self.fhandle.seek(start)
                data = self.fhandle.read(end-start)
                batch.append(np.frombuffer(data, dtype=dtype))
            return batch

    def noncontiguous_load(self, idx_values, shape):
        batch = np.empty(shape)
        for i, (idx, start, end, dtype, local_shape) in enumerate(idx_values):
            self.fhandle.seek(start)
            data = self.fhandle.read(end-start)
            batch[i] = np.frombuffer(data, dtype=dtype)
        return batch


    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            if step is not None:
                raise Exception('Step size not supported yet.')
            start, stop, total_bytes, dtype, shape, idx_values = self.get_indices(range(start, stop))
        elif isinstance(key, int):
            start, stop, total_bytes, dtype, shape, idx_values = self.get_indices(key)
        elif isinstance(key, list):
            start, stop, total_bytes, dtype, shape, idx_values = self.get_indices(key)
        elif isinstance(key, np.ndarray):
            start, stop, total_bytes, dtype, shape, idx_values = self.get_indices(key.tolist())
        else:
            raise Exception('Unsupported sice type: {0}'.format(type(key)))

        do_contiguous_load = True
        if stop-start != total_bytes:
            if stop-start > total_bytes*2:
                do_contiguous_load = False

        if do_contiguous_load:
            if self.fixed_length:
                self.fhandle.seek(stop-start)
                data = np.frombuffer(self.fhandle.read(total_bytes), dtype=dtype)
                return data.reshape(shape)
            else:
                self.fhandle.seek(stop-start)
                loaded_data = self.fhandle.read(total_bytes)
                if self.pad:
                    return self.padded_load(loaded_data, idx_values, shape[0], start)
                else:
                    return self.unpadded_load(loaded_data, idx_values, start)
        else:
            if self.fixed_length:
                return self.noncontiguous_load(idx_values, shape)
            else:
                return self.variable_length_noncontiguous_load(idx_values, shape[0])




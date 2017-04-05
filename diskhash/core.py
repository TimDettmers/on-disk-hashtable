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

def get_dill_data(name, base_path=None):
    home = base_path or join(os.environ['HOME'], '.diskhash')
    pkl_file = join(home, name + '_index.pkl')
    print(home, pkl_file, name)
    if not os.path.exists(home):
        os.mkdir(home)
    if not os.path.exists(pkl_file):
        with open(pkl_file, 'a'):
            os.utime(pkl_file, None)
        return False, pkl_file, {}
    else:
        index = dill.load(open(pkl_file))

        return True, pkl_file, index

def make_table_path(name, base_path=None):
    home = os.environ['HOME']
    base_dir = base_path or join(home, '.diskhash')
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    if not os.path.exists(join(base_dir, name)):
        with open(join(base_dir, name), 'a'):
            os.utime(join(base_dir, name), None)
    return join(base_dir, name)


class NumpyTable(object):
    def __init__(self, name, fixed_length=True, pad=True, seed=234235, base_path=None):
        self.db = None
        self.name = name
        if base_path is not None:
            self.path = join(base_path, name)
            self.index_path = join(base_path, name + '_index.pkl')
        else:
            self.path = None
            self.index_path = join(os.environ['HOME'], '.diskhash', name + '_index.pkl')
        self.fhandle = None
        self.length = 0
        self.fixed_length = fixed_length
        self.pad = pad
        self.rdm = np.random.RandomState(seed)
        self.joined_tables = []
        self.base_path = base_path

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
        tbl_exists, path, index = get_dill_data(self.name, self.base_path)
        self.db = index

        if tbl_exists:
            self.path = self.db[self.name]
            self.fhandle = open(self.path, 'r+b')
            self.length = self.db[self.name + '.length']
            self.idx = self.db[self.name + '.idx_counter']
        else:
            self.path = make_table_path(self.name, self.base_path)
            self.fhandle = open(self.path, 'r+b')
            self.db[self.name] = self.path
            self.db[self.name + '.length'] = 0
            self.db[self.name + '.idx_counter'] = 0
            self.length = 0
            self.idx = 0
            self.db[self.name + '.index'] = {}
            self.db[self.name + '.indices'] = {}
            self.db[self.name + '.index_funcs'] = {}
            self.db[self.name + '.index_length'] = {}
            self.db[self.name + '.idx2index_value'] = {}
            self.write_index()

    def add_index(self, index_name, index_func):
        assert index_name not in self.db[self.name + '.indices'], 'Index already present, cannot be overwritten!'
        self.db[self.name + '.indices'][index_name] =  {}
        self.db[self.name + '.index_funcs'][index_name] = dill.dumps(index_func)
        self.db[self.name + '.index_length'][index_name] =  {}
        self.db[self.name + '.idx2index_value'][index_name] =  {}

    def join(self, tbl):
        self.joined_tables.append(tbl)

    def __len__(self):
        return self.db[self.name + '.idx_counter']

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
        self.db[key][idx] = strvalue
        self.idx +=1
        self.db[self.name + '.idx_counter'] = self.idx

    def get_random_batch(self, batch_size, return_index=False):
        idx = self.rdm.choice(self.idx, batch_size, replace=False)
        if return_index:
            return self[idx], np.array(idx, dtype=np.int32)
        else:
            return self[idx]

    def get_random_index_batch(self, batch_size, index_name, where_indices_set=None, return_index=False):
        index = self.db[self.name + '.indices'][index_name]
        i = 0
        batches = []
        used_index = None
        while True:
            if i == 100: raise Exception('Index with a batch of size {0} is unlikely to exist!'.format(batch_size))

            index_length_dict = self.db[self.name + '.index_length'][index_name]
            index_lengths = np.array(index_length_dict.values(), dtype=np.float32)
            index_lengths = index_lengths/np.sum(index_lengths)
            choice = self.rdm.choice(len(index.keys()), 1, p=index_lengths)
            index_key = index_length_dict.keys()[choice[0]]
            if len(index[index_key]) < batch_size: continue

            if where_indices_set is not None:
                filtered_index = []
                for idx in index[index_key]:
                    if idx not in where_indices_set:
                        filtered_index.append(idx)

                if len(filtered_index) < batch_size: continue
                else:
                    idx = self.rdm.choice(filtered_index, batch_size, replace=False)
                    return idx


            if len(self.joined_tables) > 0:
                joined_idx = index[index_key]
                for tbl in self.joined_tables:
                    if index_name in tbl.db[tbl.name + '.indices']:
                        joined_idx = tbl.get_random_index_batch(batch_size, index_name, where_indices_set=joined_idx)

                for tbl in self.joined_tables:
                    batches = self[joined_idx]
                idx = joined_idx
            else:
                idx = self.rdm.choice(index[index_key], batch_size, replace=False)
                batches.append(self[idx])
            if return_index:
                if len(batches) == 1:
                    return batches[0], np.int32(idx)
                else:
                    return batches, np.int32(idx)
            else:
                if len(batches) == 1:
                    return batches[0]
                else:
                    return batches

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
            total_bytes += end-start
            if total_shape is None:
                total_shape = shape
            else:
                total_shape[0] += shape[0]

            idx_values.append([idx, start, end, dtype, shape])
        return min_start, max_end, total_bytes, dtype, total_shape, idx_values

    def append(self, nparray):
        if len(nparray.shape) == 1: nparray = nparray.reshape(1, -1)
        if len(nparray.shape) == 0: nparray = nparray.reshape(1, 1)
        for index_name in self.db[self.name + '.indices']:
            index_dict = self.db[self.name + '.indices'][index_name]
            strfunc = self.db[self.name + '.index_funcs'][index_name]
            func = dill.loads(strfunc)
            value = func(nparray)
            if value not in index_dict:
                index_dict[value] = []
                self.db[self.name + '.index_length'][index_name][value] = 0
            index_dict[value].append(self.idx)
            self.db[self.name + '.index_length'][index_name][value] += 1
            self.db[self.name + '.idx2index_value'][index_name][self.idx] = value

        self.fhandle.seek(self.length)
        start = self.fhandle.tell()
        typevalue = np2HashType[nparray.dtype].value
        bytearr = nparray.tobytes()
        self.fhandle.write(bytearr)
        end = self.fhandle.tell()
        self.length += end-start
        self.fhandle.seek(0) #this fixes a bug, but I do not know why; I do a seek(start) whenever I use the handle which should be fine
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
        batch = np.empty(shape, dtype=idx_values[0][3])
        for i, (idx, start, end, dtype, local_shape) in enumerate(idx_values):
            self.fhandle.seek(start)
            data = self.fhandle.read(end-start)
            batch[i] = np.frombuffer(data, dtype=dtype)
        return batch


    def get_index_values(self, key, index_name):
        if index_name not in self.db[self.name + '.idx2index_value']: return None

        idx2index_value = self.db[self.name + '.idx2index_value'][index_name]
        if not isinstance(key, list): key = [key]

        return np.array([idx2index_value[idx] for idx in key if idx in idx2index_value])


    def get_items(self, key, with_index_name=None):
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
            raise Exception('Unsupported slice type: {0}'.format(type(key)))

        do_contiguous_load = True
        if stop-start != total_bytes:
            if stop-start > total_bytes*2:
                do_contiguous_load = False

        ret_data = []

        if do_contiguous_load:
            if self.fixed_length:
                self.fhandle.seek(start)
                data = np.frombuffer(self.fhandle.read(stop-start), dtype=dtype)
                ret_data.append(data.reshape(shape))
            else:
                self.fhandle.seek(start)
                loaded_data = self.fhandle.read(stop-start)
                if self.pad:
                    ret_data.append(self.padded_load(loaded_data, idx_values, shape[0], start))
                else:
                    ret_data.append(self.unpadded_load(loaded_data, idx_values, start))
        else:
            if self.fixed_length:
                ret_data.append(self.noncontiguous_load(idx_values, shape))
            else:
                ret_data.append(self.variable_length_noncontiguous_load(idx_values, shape[0]))

        if with_index_name is not None:
            idx = self.get_index_values(key, with_index_name)
            data = ret_data.pop(0)
            if idx is not None:
                ret_data.append([idx, data])
            else:
                ret_data.append([data])

        if len(self.joined_tables) > 0:
            for tbl in self.joined_tables:
                ret_data.append(tbl.get_items(key, with_index_name))


            return ret_data
        else:
            return ret_data[0]


    def __getitem__(self, key):
         return self.get_items(key)

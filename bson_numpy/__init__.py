import ctypes
from ctypes import *
from ctypes.util import find_library

import numpy as np


class bson_error_t(Structure):
    _fields_ = [('domain', c_uint32),
                ('code', c_uint32),
                ('msg', c_char * 504)]

bson_error_ptr = POINTER(bson_error_t)

bson_t = c_byte * 512
bson_ptr = POINTER(bson_t)
bson_iter_t = c_byte * 80
bson_iter_ptr = POINTER(bson_iter_t)

libbson = cdll.LoadLibrary(find_library(
    "/Users/emptysquare/.virtualenvs/c-driver/libbson/.libs/libbson-1.0.0.dylib"
))

libbson.bson_strdup.argtypes = [c_char_p]
libbson.bson_strdup.restype = c_char_p

libbson.bson_new.restype = bson_ptr

for type_name in ['int32']:
    ctypes_type = getattr(ctypes, 'c_' + type_name)

    # E.g., bool bson_append_int32 (bson_t *, char *key, int key_len, int32_t).
    append = getattr(libbson, 'bson_append_' + type_name)
    append.argtypes = [bson_ptr, c_char_p, ctypes_type]
    append.restype = c_bool

    # E.g., int32_t bson_iter_int32 (bson_iter_t *).
    get = getattr(libbson, 'bson_iter_' + type_name)
    get.argtypes = [bson_iter_ptr]
    get.restype = ctypes_type


libbson.bson_as_json.argtypes = [bson_ptr, POINTER(c_size_t)]
libbson.bson_as_json.restype = c_char_p

libbson.bson_new_from_json.argtypes = [c_char_p, c_ssize_t, bson_error_ptr]
libbson.bson_new_from_json.restype = bson_ptr

libbson.bson_destroy.argtypes = [bson_ptr]
libbson.bson_destroy.restype = None

libbson.bson_iter_init.argtypes = [bson_iter_ptr, bson_ptr]
libbson.bson_iter_init.restype = c_bool

libbson.bson_iter_key.argtypes = [bson_iter_ptr]
libbson.bson_iter_key.restype = c_char_p

# TODO: Decimal128, plus other types defined in Monary but not here
NUMPY_TYPES = {
    'objectid':  np.dtype('<V12'),  # TODO: modern way to express this?
    'bool':      np.bool,
    'int8':      np.int8,
    'int16':     np.int16,
    'int32':     np.int32,
    'int64':     np.int64,
    'uint8':     np.uint8,
    'uint16':    np.uint16,
    'uint32':    np.uint32,
    'uint64':    np.uint64,
    'float32':   np.float32,
    'float64':   np.float64,
    'date':      np.int64,
    'timestamp': np.uint64,
}

BSON_TYPES = dict([v, k] for k, v in NUMPY_TYPES.items())


# TODO: bson_buffer should be bytes, do bson_init_from_buffer here
def load(bson_buffer, dtype, fields=None):
    ret = []
    dtype = np.dtype(dtype)  # Convert from list of tuples if necessary.
    iter_p = pointer(bson_iter_t())
    libbson.bson_iter_init(iter_p, bson_buffer)
    while libbson.bson_iter_next(iter_p):
        field = libbson.bson_iter_key(iter_p)
        field_type = dtype.fields[field][0]
        fn = getattr(libbson, 'bson_iter_' + BSON_TYPES[field_type.type])
        ret.append(fn(iter_p))
    return np.array([tuple(ret)], dtype=dtype)

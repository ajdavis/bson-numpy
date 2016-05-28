import ctypes
from ctypes import *
from ctypes.util import find_library

import numpy as np
from numpy import ma


class bson_error_t(Structure):
    _fields_ = [('domain', c_uint32),
                ('code', c_uint32),
                ('msg', c_char * 504)]

bson_error_ptr = POINTER(bson_error_t)


class bson_t(Structure):
    _fields_ = [('flags', c_uint32),
                ('len', c_uint32),
                ('padding', c_byte * 120)]

bson_ptr = POINTER(bson_t)
bson_iter_t = c_byte * 80
bson_iter_ptr = POINTER(bson_iter_t)


class bson_writer_t(Structure):
    pass

bson_writer_ptr = POINTER(bson_writer_t)


class bson_reader_t(Structure):
    pass

bson_reader_ptr = POINTER(bson_reader_t)

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

libbson.bson_iter_type.argtypes = [bson_iter_ptr]
libbson.bson_iter_type.restype = c_int

libbson.bson_get_data.argtypes = [bson_ptr]
libbson.bson_get_data.restype = POINTER(c_uint8)

libbson.bson_new_from_data.argtypes = [POINTER(c_uint8), c_size_t]
libbson.bson_new_from_data.restype = bson_ptr

libbson.bson_new_from_buffer.argtypes = [POINTER(POINTER(c_uint8)),
                                         POINTER(c_size_t),
                                         c_void_p,
                                         c_void_p]
libbson.bson_new_from_buffer.restype = bson_ptr

libbson.bson_init_static.argtypes = [bson_ptr, POINTER(c_uint8), c_size_t]
libbson.bson_init_static.restype = c_bool

libbson.bson_reader_new_from_data.argtypes = [POINTER(c_uint8), c_size_t]
libbson.bson_reader_new_from_data.restype = bson_reader_ptr

libbson.bson_reader_read.argtypes = [bson_reader_ptr, POINTER(c_bool)]
libbson.bson_reader_read.restype = bson_ptr

libbson.bson_writer_new.argtypes = [POINTER(POINTER(c_uint8)),
                                    POINTER(c_size_t),
                                    c_size_t,
                                    c_void_p,
                                    c_void_p]
libbson.bson_writer_new.restype = bson_writer_ptr

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


def from_bson_buffer(buf, buf_len, dtype, fields=None):
    """Convert from buffer of catenated BSON documents to NumPy array."""
    ret = []
    mask = []
    dtype = np.dtype(dtype)  # Convert from list of tuples if necessary.
    field_offsets = dict((field, i) for i, field in enumerate(dtype.fields))
    it = byref(bson_iter_t())

    eof = c_bool()
    reader = libbson.bson_reader_new_from_data(buf, buf_len)

    assert reader
    b = libbson.bson_reader_read(reader, byref(eof))
    while b:
        assert libbson.bson_iter_init(it, b)
        row = []
        row_mask = []
        for field, field_type in dtype.fields.items():
            # All entries in this row are masked out to begin.
            row.append(field_type[0].type())
            row_mask.append(1)

        while libbson.bson_iter_next(it):
            field = libbson.bson_iter_key(it)
            if field in dtype.fields:
                field_type = dtype.fields[field][0]
                fn = getattr(libbson, 'bson_iter_' + BSON_TYPES[field_type.type])
                row[field_offsets[field]] = fn(it)
                row_mask[field_offsets[field]] = 0

        ret.append(tuple(row))
        mask.append(tuple(row_mask))
        b = libbson.bson_reader_read(reader, byref(eof))

    return ma.array(ret, mask=mask, dtype=dtype)


def from_bson_array(buf, buf_len, dtype, fields=None):
    """Convert from BSON array like {"0": doc, "1": doc, ...} to NumPy array.

    The MongoDB "find" command and others return batches of documents like:

        {"firstBatch": {"0": doc, "1": doc, ...}}

    Or, from the "getMore" command:

        {"nextBatch": {"0": doc, "1": doc, ...}}

    The batch element is a BSON array, which is like a document whose keys are
    ASCII decimal numbers. Pull each document from the array and add its fields
    the resulting NumPy array, converted according to "dtype".
    """
    ret = []
    mask = []
    dtype = np.dtype(dtype)  # Convert from list of tuples if necessary.
    field_offsets = dict((field, i) for i, field in enumerate(dtype.fields))

    bson_array_doc = bson_t()
    assert libbson.bson_init_static(byref(bson_array_doc), buf, buf_len)

    array_it = byref(bson_iter_t())
    assert libbson.bson_iter_init(array_it, byref(bson_array_doc))

    it = byref(bson_iter_t())

    while libbson.bson_iter_next(array_it):
        assert libbson.bson_iter_type(array_it) == 0x3  # BSON document.
        row = []
        row_mask = []
        for field, field_type in dtype.fields.items():
            # All entries in this row are masked out to begin.
            row.append(field_type[0].type())
            row_mask.append(1)

        assert libbson.bson_iter_recurse(array_it, it)
        while libbson.bson_iter_next(it):
            field = libbson.bson_iter_key(it)
            if field in dtype.fields:
                field_type = dtype.fields[field][0]
                fn = getattr(libbson, 'bson_iter_' + BSON_TYPES[field_type.type])
                row[field_offsets[field]] = fn(it)
                row_mask[field_offsets[field]] = 0

        ret.append(tuple(row))
        mask.append(tuple(row_mask))

    return ma.array(ret, mask=mask, dtype=dtype)

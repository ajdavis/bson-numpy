import unittest
from ctypes import *

import numpy as np
import numpy.testing as nptest
from numpy import ma

import bson_numpy as bn
from bson_numpy import libbson


def create_buffer(docs):
    """Catenate a sequence of BSON byte buffers into one."""
    buf_len = sum(doc.contents.len for doc in docs)
    buf = cast(create_string_buffer(buf_len), POINTER(c_uint8))
    offset = 0
    for doc in docs:
        # Unfortunately hard: http://bugs.python.org/issue6259.
        buf_ptr = cast(addressof(buf.contents) + offset, POINTER(c_uint8))
        memmove(buf_ptr,
                libbson.bson_get_data(doc),
                c_size_t(doc.contents.len))
        offset += doc.contents.len

    return buf, buf_len


class BSONNumPyTest(unittest.TestCase):
    def setUp(self):
        doc0 = libbson.bson_new()
        libbson.bson_append_int32(doc0, 'a', -1, 1)
        libbson.bson_append_int32(doc0, 'b', -1, 2)

        # Lacks 'a', adds 'c'.
        doc1 = libbson.bson_new()
        libbson.bson_append_int32(doc1, 'b', -1, 3)
        libbson.bson_append_int32(doc1, 'c', -1, 4)
        self.docs = [doc0, doc1]
        self.dtype = [('a', np.int32), ('b', np.int32)]

        # The '1' in the mask marks a missing value.
        self.expected_array = ma.array([(1, 2), (0, 3)],
                                       mask=[(0, 0), (1, 0)],
                                       dtype=self.dtype)

    def assert_array_eq(self, actual, expected, err_msg=''):
        nptest.assert_array_equal(actual, expected, err_msg=err_msg)
        self.assertEqual(actual.dtype, expected.dtype)

    def test_bson_buffer(self):
        buf, buf_len = create_buffer(self.docs)
        self.assert_array_eq(bn.from_bson_buffer(buf, buf_len, self.dtype),
                             self.expected_array)

    def test_bson_array(self):
        bson_array = libbson.bson_new()
        for i, doc in enumerate(self.docs):
            assert libbson.bson_append_document(bson_array, str(i), -1, doc)

        buf = libbson.bson_get_data(bson_array)
        self.assert_array_eq(
            bn.from_bson_array(buf, bson_array.contents.len, self.dtype),
            self.expected_array)

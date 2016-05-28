import unittest

import numpy as np
import numpy.testing as nptest

import bson_numpy as bn
from bson_numpy import libbson


class BSONNumPyTest(unittest.TestCase):
    def assert_array_eq(self, actual, expected, err_msg=''):
        nptest.assert_array_equal(actual, expected, err_msg=err_msg)
        self.assertEqual(actual.dtype, expected.dtype)

    def test_int32(self):
        dtype = [('a', np.int32), ('b', np.int32)]
        expected = np.array([(1, 2)], dtype=dtype)
        doc = libbson.bson_new()
        libbson.bson_append_int32(doc, 'a', -1, 1)
        libbson.bson_append_int32(doc, 'b', -1, 2)
        self.assert_array_eq(bn.load(doc, dtype), expected)
        libbson.bson_destroy(doc)

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy.testing as nptest


class BSONNumPyTestBase(unittest.TestCase):
    def assert_array_eq(self, actual, expected, err_msg=''):
        nptest.assert_array_equal(actual, expected, err_msg=err_msg)
        self.assertEqual(actual.dtype, expected.dtype)

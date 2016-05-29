from ctypes import *

import numpy as np
import pymongo
from bson.codec_options import CodecOptions
from bson.raw_bson import RawBSONDocument
from numpy import ma

import bson_numpy
from tests import BSONNumPyTestBase

opts = CodecOptions(document_class=RawBSONDocument)
dtype = np.dtype([('_id', np.int32), ('a', np.float)])

# TODO: test masking
expected = ma.array([(1, 1.0)], dtype=dtype)


class TestPyMongo(BSONNumPyTestBase):
    @classmethod
    def setUpClass(cls):
        cls.client = pymongo.MongoClient()
        cls.db = cls.client.test
        # Ensure a document.
        cls.db.collection.update_one({'_id': 1},
                                     {'$set': {'a': 1.0}},
                                     upsert=True)

        cls.collection = cls.db.get_collection('collection', codec_options=opts)
        # cls.wire_version = cls.db.command('ismaster')['maxWireVersion']

    @classmethod
    def tearDownClass(cls):
        cls.collection = cls.db = cls.client = None

    def test_find_one(self):
        raw = self.collection.find_one(1).raw
        buf = cast(create_string_buffer(raw), POINTER(c_uint8))
        np_array = bson_numpy.from_bson_buffer(buf, len(raw), dtype)
        self.assert_array_eq(np_array, expected)

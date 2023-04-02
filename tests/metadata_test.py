from collections import OrderedDict
import unittest
import copy

from model_util import md_obj, md_obj_fields
import base_model

class Obj(base_model.BaseModel):
    _metadata_fields = 'scalar1 scalar2'.split()
    _model_fields = _metadata_fields.copy()

    scalar1: any
    scalar2: any
    def __init__(self, scalar1: any, scalar2: any):
        self.scalar1 = scalar1
        self.scalar2 = scalar2

class TestGen(unittest.TestCase):
    def test_tuple_list(self):
        tuple_out = md_obj((1, 2, 3))
        list_out = md_obj([1, 2, 3])
        self.assertEqual(tuple_out, list_out)

    def test_inner_tuple(self):
        out = md_obj(
            dict(
                one="one",
                two="two",
                innertuple=(1, 2, 3)
            )
        )
        expected = dict(
            one="one",
            two="two",
            innertuple=[1, 2, 3]
        )

        self.assertEqual(expected, out)

    def test_dict_order(self):
        one = {'one': 1, 'two': 2, 'three': 3, 'four': 4}
        two = {'four': 4, 'three': 3, 'two': 2, 'one': 1}
        one_out = md_obj(one)
        two_out = md_obj(two)

        self.assertEqual(one_out, two_out)
    
    def test_contents_dict(self):
        inputs = dict(
            scalar1="one",
            scalar2="two",
            dict1=dict(
                subscalar1="ONE",
                subscalar2="TWO"
            )
        )

        expected = copy.deepcopy(inputs)
        actual = md_obj(inputs)

        self.assertDictEqual(expected, actual)

    def test_contents_OrderedDict(self):
        inputs = dict(
            scalar1="one",
            scalar2="two",
            dict1=OrderedDict(
                subscalar1="ONE",
                subscalar2="TWO"
            )
        )

        expected = copy.deepcopy(inputs)
        actual = md_obj(inputs)

        self.assertDictEqual(expected, actual)

    def test_contents_Obj(self):
        inputs = dict(
            top1="one",
            top2="two",
            obj1=Obj(scalar1="sub1", scalar2="sub2")
        )

        expected = dict(
            top1="one",
            top2="two",
            obj1=dict(
                scalar1="sub1",
                scalar2="sub2",
            )
        )
        actual_fields = md_obj_fields(inputs)
        actual = md_obj(inputs)

        for expect_key in 'top1 top2 obj1'.split():
            self.assertIn(expect_key, actual_fields)
            self.assertIn(expect_key, actual)

        self.assertDictEqual(expected, actual)

import unittest
import tempfile
from typing import Mapping, Any

from experiment import Experiment
import base_model
import checkpoint_util
from pathlib import Path
import torch
from torch import nn
import train_util

class DumbNet(base_model.BaseModel):
    _metadata_fields = 'one two'.split()
    _model_fields = 'one two p'.split()

    def __init__(self, one: any, two: any):
        super().__init__()
        self.one = one
        self.two = two
        self.p = nn.Parameter(torch.zeros((4, 4)))

def odict2dict(obj: any) -> any:
    if isinstance(obj, dict):
        res = dict()
        for field, val in obj.items():
            res[field] = odict2dict(val)
        return res
    elif isinstance(obj, list):
        return [odict2dict(item) for item in obj]
    return obj

class TestBase(unittest.TestCase):
    def assertDictEqual(self, d1: Mapping[Any, object], d2: Mapping[Any, object], msg: Any = None) -> None:
        d1 = odict2dict(d1)
        d2 = odict2dict(d2)
        return super().assertDictEqual(d1, d2, msg)
    
    def assertDictContainsSubset(self, subset: Mapping[Any, Any], dictionary: Mapping[Any, Any], msg: object = None) -> None:
        subset = odict2dict(subset)
        dictionary = odict2dict(dictionary)
        return super().assertDictContainsSubset(subset, dictionary, msg)

class TestIdentity(TestBase):
    def test_simple(self):
        exp1 = Experiment(label="foo")
        exp2 = Experiment(label="foo")

        self.assertEqual(exp1.shortcode, exp2.shortcode)

    def test_simple_notsame(self):
        exp1 = Experiment(label="foo")
        exp2 = Experiment(label="bar")

        self.assertNotEqual(exp1.shortcode, exp2.shortcode)
    
    def test_net_same(self):
        exp1 = Experiment(label="foo", net=DumbNet(one=1, two=2))
        exp2 = Experiment(label="foo", net=DumbNet(one=1, two=2))

        self.assertEqual(exp1.shortcode, exp2.shortcode)
    
    def test_net_notsame(self):
        exp1 = Experiment(label="foo", net=DumbNet(one=1, two=2))
        exp2 = Experiment(label="foo", net=DumbNet(one=11, two=2))

        self.assertNotEqual(exp1.shortcode, exp2.shortcode)
    
    def test_id_fields(self):
        exp1 = Experiment(label="foo", net=DumbNet(one=1, two=2))
        expected = ['label', 'net_args']
        actual = exp1.id_fields()
        self.assertEqual(expected, actual)

    def test_id_values(self):
        exp1 = Experiment(label="foo", net=DumbNet(one=1, two=2))
        expected = dict(
            label='foo',
            net_args=dict(
                one=1,
                two=2,
            )
        )
        expected['net_args']['class'] = 'DumbNet'
        actual = exp1.id_values()
        self.assertDictContainsSubset(expected, actual)

    def test_id_loadsave(self):
        self.maxDiff = None

        exp1 = Experiment()
        exp1.net = DumbNet(one=1, two=2)
        exp1_md = exp1.metadata_dict()
        exp1_fields = exp1.id_fields()
        exp1_values = exp1.id_values()

        exp2 = Experiment().load_model_dict(exp1_md)
        exp2_fields = exp2.id_fields()
        exp2_values = exp2.id_values()

        self.assertEqual(exp1_fields, exp2_fields)
        self.assertEqual(dict(exp1_values), dict(exp2_values))



class TestMetaBackcompat(TestBase):
    def test_global_nepochs(self):
        md = {
            'nepochs': 10
        }
        exp = Experiment().load_model_dict(md)
        self.assertEqual(10, exp.cur_run().nepochs)
        self.assertEqual(10, exp.nepochs)
    
    def test_global_nepochs(self):
        md = {
            "resumed_at": [
                {"nepochs": 5,
                 "timestamp": "2023-03-23 21:13:48",
                 "path": "runs/ae_conv_vae_new_005_20230323-200328/checkpoints/k3-s1-128-s1-128x2-s2-128-s1-256x2-s2-256-s1-512x2-s2-512-s1-512x2-8,enc_kern_3,latdim_8_64_64,ratio_0.042,image_size_512,inl_silu,fnl_sigmoid,loss_edge+l2_sqrt+kl,epoch_0005.ckpt"
                }
            ],
            "nepochs": 21,
        }
        exp = Experiment().load_model_dict(md)
        self.assertEqual(21, exp.cur_run().nepochs)
        self.assertEqual(21, exp.nepochs)

class TestSaveNet(TestBase):
    def test_save(self):
        exp = Experiment()
        exp.net = DumbNet(one=1, two=2)
        
        md = exp.metadata_dict()
        self.assertEqual(1, md['net_args']['one'])
        self.assertEqual(2, md['net_args']['two'])

class TestLoad(TestBase):
    def test_load_same_meta(self):
        self.maxDiff = None

        exp1 = Experiment()
        exp1.net = DumbNet(one=1, two=2)
        exp1_md = exp1.metadata_dict()

        exp2 = Experiment().load_model_dict(exp1_md)
        exp2_md = exp2.metadata_dict()
        self.assertDictEqual(dict(exp1_md), dict(exp2_md))

    def test_load_meta(self):
        exp1 = Experiment()
        exp1.net = DumbNet(one=1, two=2)
        exp1_md = exp1.metadata_dict()

        exp2 = Experiment().load_model_dict(exp1_md)
        self.assertEquals(1, exp2.net_one)
        self.assertEquals(2, exp2.net_two)

    def test_shortcode(self):
        exp = Experiment()
        exp.net = DumbNet(one=1, two=2)
        exp1_shortcode = exp.shortcode

        exp2 = Experiment().load_model_dict(exp.metadata_dict())
        exp2_shortcode = exp2.shortcode

        if exp1_shortcode != exp2_shortcode:
            diff_fields = exp.id_compare(exp2)
            print("diff_fields:", " ".join(map(str, diff_fields)))
        self.assertEqual(exp1_shortcode, exp2_shortcode)

    def test_shortcode_model(self):
        exp = Experiment()
        exp.net = DumbNet(one=1, two=2)
        exp.startlr = 1.0e-4
        exp.endlr = 1.0e-5
        exp.optim = train_util.lazy_optim_fn(exp)
        exp.sched = train_util.lazy_sched_fn(exp)
        shortcode = exp.shortcode

        with tempfile.NamedTemporaryFile("w") as file:
            ckpt_path = Path(str(file) + ".ckpt")
            json_path = Path(str(file) + ".json")
            checkpoint_util.save_ckpt_and_metadata(exp, ckpt_path, json_path)

            state_dict = torch.load(ckpt_path)

        exp_load = Experiment().load_model_dict(state_dict)
        print("diff fields:")
        print(exp_load.id_compare(exp))
        self.assertEqual(shortcode, exp_load.shortcode)
        # NOTE: can't compare net_one, net_two, as the 'net' isn't actually
        # instantiated.

    def test_netGet(self):
        exp = Experiment()
        exp.net = DumbNet(one=1, two=2)

        self.assertEquals(1, exp.net_one)
        self.assertEqual(2, exp.net_two)
        self.assertEqual('DumbNet', exp.net_class)

    def test_netGet_onLoad(self):
        exp = Experiment()
        exp.net = DumbNet(one=1, two=2)
        exp = Experiment().load_model_dict(exp.metadata_dict())

        self.assertEquals(1, exp.net_one)
        self.assertEqual(2, exp.net_two)
        self.assertEqual('DumbNet', exp.net_class)

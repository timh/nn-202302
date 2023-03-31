import unittest
import tempfile

from experiment import Experiment
import base_model
import checkpoint_util
from pathlib import Path
import torch
from torch import nn
import train_util

usually_same = set('sched_type sched_warmup_epochs do_compile skip '
                   'batch_size optim_type max_epochs net_class finished'.split())
class TestShortcode(unittest.TestCase):
    def test_simple(self):
        exp1 = Experiment(label="foo")
        exp2 = Experiment(label="foo")

        self.assertEqual(exp1.shortcode, exp2.shortcode)

    def test_simple_notsame(self):
        exp1 = Experiment(label="foo")
        exp2 = Experiment(label="bar")

        self.assertNotEqual(exp1.shortcode, exp2.shortcode)

class TestMetaBackcompat(unittest.TestCase):
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

class DumbNet(base_model.BaseModel):
    _metadata_fields = 'one two'.split()
    _model_fields = 'one two p'.split()

    def __init__(self, one: int, two: int):
        super().__init__()
        self.one = one
        self.two = two
        self.p = nn.Parameter(torch.zeros((4, 4)))

class TestSaveNet(unittest.TestCase):
    def test_save(self):
        exp = Experiment()
        exp.net = DumbNet(one=1, two=2)
        
        md = exp.metadata_dict()
        self.assertEqual(1, md['net_args']['one'])
        self.assertEqual(2, md['net_args']['two'])

class TestLoad(unittest.TestCase):
    def test_shortcode(self):
        exp = Experiment()
        exp.net = DumbNet(one=1, two=2)
        exp1_shortcode = exp.shortcode

        exp2 = Experiment().load_model_dict(exp.metadata_dict())
        exp2_shortcode = exp2.shortcode

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

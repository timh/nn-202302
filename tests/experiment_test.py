from pathlib import Path

import torch

from .base import TestBase, DumbNet
from experiment import Experiment
import train_util
import model_util

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
        expected = ['label', 'loss_type', 'net_args']
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
        self.assertEqual(10, exp.nepochs)
    
    def test_run_nepochs(self):
        md = {
            "runs": [
                {"nepochs": 5,
                 "created_at": "2023-03-23 21:13:48",
                },
                {"nepochs": 25,
                 "created_at": "2023-03-23 21:13:48",
                },
                {"nepochs": 15,
                 "created_at": "2023-03-23 21:13:48",
                }
            ],
        }
        exp = Experiment().load_model_dict(md)
        self.assertEqual(15, exp.nepochs)

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

        _json_path, ckpt_path = self.save_checkpoint(exp)
        state_dict = torch.load(ckpt_path)

        exp_load = Experiment().load_model_dict(state_dict)
        print("diff fields:")
        print(exp_load.id_diff(exp))
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

class TestRuns(TestBase):
    def test_init(self):
        exp = Experiment()

        # get_run() will lazily instantiate the first run.
        run = exp.get_run()
        self.assertEquals(1, len(exp.runs))

        self.assertEqual(0, run.checkpoint_nepochs)
        self.assertEqual(0, run.checkpoint_nbatches)
        self.assertEqual(0, run.checkpoint_nsamples)
        self.assertEqual(None, run.checkpoint_path)
    
    def test_roundtrip(self):
        exp = Experiment()

        # get_run() will lazily instantiate the first run.
        run = exp.get_run()
        run.checkpoint_nepochs = 10
        run.checkpoint_path = Path("foo")

        print("test_roundtrip:")
        model_util.print_dict(exp.metadata_dict())

        exp = Experiment().load_model_dict(exp.metadata_dict())
        self.assertEqual(1, len(exp.runs))

        run = exp.get_run()
        self.assertEqual(10, run.checkpoint_nepochs)
        self.assertEqual(Path("foo"), run.checkpoint_path)
    
    def test_load(self):
        md = dict(
            runs=[
                dict(checkpoint_nepochs=10)
            ]
        )
        print("test_load: manual md:")
        model_util.print_dict(md)

        exp = Experiment().load_model_dict(md)

        self.assertEqual(1, len(exp.runs))

        run = exp.get_run()
        self.assertEqual(10, run.checkpoint_nepochs)

class TestHist(TestBase):
    def test_loss_hist(self):
        # setup
        exp = self.create_dumb_exp()
        exp.train_loss_hist = [(0.1) for _ in range(10)]
        exp.val_loss_hist = [(epoch, 0.1) for epoch in range(10)]
        exp_md = exp.metadata_dict()

        # execute
        loaded = Experiment().load_model_dict(exp_md)

        # assert
        self.assertEqual(10, len(loaded.train_loss_hist))
        self.assertEqual(10, len(loaded.val_loss_hist))

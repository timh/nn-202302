from pathlib import Path
import datetime

import torch

from .base import TestBase, DumbNet
from nnexp.experiment import Experiment
from nnexp.utils import model_util
from nnexp.training import train_util

class TestIdentity(TestBase):
    def test_simple(self):
        exp1 = Experiment(label="foo")
        exp2 = Experiment(label="foo")

        assert exp2.shortcode == exp1.shortcode

    def test_simple_notsame(self):
        exp1 = Experiment(label="foo")
        exp2 = Experiment(label="bar")

        assert exp2.shortcode != exp1.shortcode
    
    def test_net_same(self):
        exp1 = Experiment(label="foo", net=DumbNet(one=1, two=2))
        exp2 = Experiment(label="foo", net=DumbNet(one=1, two=2))

        assert exp2.shortcode == exp1.shortcode
    
    def test_net_notsame(self):
        exp1 = Experiment(label="foo", net=DumbNet(one=1, two=2))
        exp2 = Experiment(label="foo", net=DumbNet(one=11, two=2))

        assert exp2.shortcode != exp1.shortcode
    
    def test_id_fields(self):
        exp1 = Experiment(label="foo", net=DumbNet(one=1, two=2))
        expected = ['label', 'loss_type', 'net_args']
        actual = exp1.id_fields()

        assert actual == expected

    def test_id_values(self):
        exp1 = Experiment(label="foo", net=DumbNet(one=1, two=2))
        expected = dict(
            label='foo',
            net_args=dict(
                one=1,
                two=2,
            ),
            loss_type='l2'
        )
        expected['net_args']['class'] = 'DumbNet'
        actual = exp1.id_values()

        assert (actual | expected) == expected

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

        assert exp1_fields == exp2_fields
        assert dict(exp1_values) == dict(exp2_values)

class TestMetaBackcompat(TestBase):
    def test_global_nepochs(self):
        md = {
            'nepochs': 10
        }
        exp = Experiment().load_model_dict(md)
        assert exp.nepochs == 10
    
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
        assert exp.nepochs == 15

class TestSaveNet(TestBase):
    def test_save(self):
        exp = Experiment()
        exp.net = DumbNet(one=1, two=2)
        
        md = exp.metadata_dict()
        assert md['net_args']['one'] == 1
        assert md['net_args']['two'] == 2

class TestLoad(TestBase):
    def test_shortcode_created_at(self):
        exp1 = Experiment(label="foo")
        exp1.created_at = datetime.datetime.strptime("2023-01-01", "%Y-%m-%d")
        exp1.something_else = "bar"
        exp2 = Experiment().load_model_dict(exp1.metadata_dict())

        assert exp1.shortcode == exp2.shortcode
        assert exp1.created_at == exp2.created_at

class TestLoadNet(TestBase):
    def test_load_same_meta(self):
        self.maxDiff = None

        exp1 = Experiment()
        exp1.net = DumbNet(one=1, two=2)
        exp1_md = exp1.metadata_dict()

        exp2 = Experiment().load_model_dict(exp1_md)
        exp2_md = exp2.metadata_dict()

        assert dict(exp1_md) == dict(exp2_md)

    def test_load_meta(self):
        exp1 = Experiment()
        exp1.net = DumbNet(one=1, two=2)
        exp1_md = exp1.metadata_dict()

        exp2 = Experiment().load_model_dict(exp1_md)
        assert exp2.net_one == 1
        assert exp2.net_two == 2

    def test_shortcode(self):
        exp1 = Experiment()
        exp1.net = DumbNet(one=1, two=2)
        exp1_shortcode = exp1.shortcode

        exp2 = Experiment().load_model_dict(exp1.metadata_dict())
        exp2_shortcode = exp2.shortcode

        if exp1_shortcode != exp2_shortcode:
            diff_fields = exp1.id_compare(exp2)
            print("diff_fields:", " ".join(map(str, diff_fields)))

        assert exp1_shortcode == exp2_shortcode

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

        assert exp_load.shortcode == shortcode
        # NOTE: can't compare net_one, net_two, as the 'net' isn't actually
        # instantiated.

    def test_netGet(self):
        exp = Experiment()
        exp.net = DumbNet(one=1, two=2)

        assert exp.net_one == 1
        assert exp.net_two == 2

        assert exp.net_class == 'DumbNet'

    def test_netGet_onLoad(self):
        exp = Experiment()
        exp.net = DumbNet(one=1, two=2)
        exp = Experiment().load_model_dict(exp.metadata_dict())

        assert exp.net_one == 1
        assert exp.net_two == 2
        assert exp.net_class == 'DumbNet'

class TestRuns(TestBase):
    def test_init(self):
        exp = Experiment()

        # get_run() will lazily instantiate the first run.
        run = exp.get_run()
        assert len(exp.runs) == 1

        assert run.checkpoint_nepochs == 0
        assert run.checkpoint_nbatches == 0
        assert run.checkpoint_nsamples == 0
        assert run.checkpoint_path == None
    
    def test_roundtrip(self):
        exp = Experiment()

        # get_run() will lazily instantiate the first run.
        run = exp.get_run()
        run.checkpoint_nepochs = 10
        run.checkpoint_path = Path("foo")

        print("test_roundtrip:")
        model_util.print_dict(exp.metadata_dict())

        exp = Experiment().load_model_dict(exp.metadata_dict())
        assert len(exp.runs) == 1

        run = exp.get_run()
        assert run.checkpoint_nepochs == 10
        assert run.checkpoint_path == Path("foo")
    
    def test_load(self):
        md = dict(
            runs=[
                dict(checkpoint_nepochs=10)
            ]
        )
        print("test_load: manual md:")
        model_util.print_dict(md)

        exp = Experiment().load_model_dict(md)

        assert len(exp.runs) == 1

        run = exp.get_run()
        assert run.checkpoint_nepochs == 10

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
        assert len(loaded.train_loss_hist) == 10
        assert len(loaded.val_loss_hist) == 10

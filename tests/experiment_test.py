import unittest
import sys

from experiment import Experiment

usually_same = set('sched_type sched_warmup_epochs do_compile skip '
                   'batch_size optim_type max_epochs net_class finished'.split())
class TestExperiment(unittest.TestCase):
    def test_simple_same(self):
        exp1 = Experiment(label="foo")
        exp2 = Experiment(label="foo")

        same, fields_same, fields_diff = exp1.is_same(exp2, return_tuple=True)
        self.assertEqual(True, same)
        self.assertEqual({'label'} | usually_same, fields_same)
        self.assertEqual(set(), fields_diff)

    def test_simple_notsame(self):
        exp1 = Experiment(label="foo")
        exp2 = Experiment(label="bar")

        same, fields_same, fields_diff = exp1.is_same(exp2, return_tuple=True)
        print("fields_same: " + ",".join(fields_same))
        print("fields_diff: " + ",".join(fields_diff))
        self.assertEqual(False, same)
        self.assertEqual(usually_same, fields_same)
        self.assertEqual({'label'}, fields_diff)
    
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

    def test_global_nepochs2(self):
        from pathlib import Path
        import json
        path = Path("/home/tim/devel/nn-202302/denoise/runs",
                    "ae_conv_vae_new_050_20230323-211348",
                    "checkpoints",
                    "k3-s1-128-s1-128x2-s2-128-s1-256x2-s2-256-s1-512x2-s2-512-s1-512x2-8,enc_kern_3,latdim_8_64_64,ratio_0.042,image_size_512,inl_silu,fnl_sigmoid,loss_edge+l2_sqrt+kl,epoch_0021.json")
        with open(path, "r") as file:
            md = json.load(file)
        exp = Experiment().load_model_dict(md)
        self.assertEqual(21, exp.cur_run().nepochs)
        self.assertEqual(21, exp.nepochs)
        
        md_out = exp.metadata_dict()
        self.assertEquals(21, md_out['nepochs'])

        

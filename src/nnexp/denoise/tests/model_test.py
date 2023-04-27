import sys
import unittest
from typing import List
from pathlib import Path

import torch
from torch import Tensor

sys.path.append("..")
import conv_types as ct
from denoise.models import vae

class TestConvConfig(unittest.TestCase):
    def test_simple(self):
        cfg = ct.make_config(layers_str="k3-s1-128-64-32", in_chan=3, in_size=512)
        self.assertEqual(3, len(cfg.layers))
        self.assertEqual([128, 64, 32], [l.out_chan('down') for l in cfg.layers])
        self.assertEqual([3, 128, 64], [l.out_chan('up') for l in cfg.layers])

    def test_simple2(self):
        cfg = ct.make_config(layers_str="k3-s1-256x2-s2-256-s1-128x2-s2-128-s1-64x2-s2-64-s1-8", in_chan=3, in_size=512)
        self.assertEqual([512, 512, 256, 256, 128, 128, 64, 64], [l.out_size('down') for l in cfg.layers])
    

    def test_metadata_dict(self):
        cfg = ct.make_config(layers_str="k3-s1-128-64-32", in_chan=3, in_size=512)
        expected = dict(
            layers_str="k3-128-64-32",
            inner_nl_type='relu',
            linear_nl_type='relu',
            final_nl_type='sigmoid',
            inner_norm_type='layer',
            final_norm_type='layer',
            norm_num_groups=32
        )
        self.assertEqual(expected, cfg.metadata_dict())
    
    def test_channel_repeat(self):
        cfg = ct.make_config(layers_str="k3-s1-128x3", in_chan=3, in_size=512)
        self.assertEqual('k3-128x3', cfg.metadata_dict()['layers_str'])

    def test_channel_repeat_summarize(self):
        cfg = ct.make_config(layers_str="k3-s1-128-128-128", in_chan=3, in_size=512)
        self.assertEqual('k3-128x3', cfg.metadata_dict()['layers_str'])

    def test_channel_repeat_fancy(self):
        cfg = ct.make_config(layers_str="k3-s1-128-128-128s2-128-128", in_chan=3, in_size=512)
        self.assertEqual('k3-128x2-128s2-128x2', cfg.metadata_dict()['layers_str'])

    def test_sizes_down(self):
        cfg = ct.make_config(layers_str="k3-256x2-256s2-128x2-128s2-64x2-64s2-8", in_chan=3, in_size=512)

import sys
import unittest
from typing import List
from pathlib import Path

import torch
from torch import Tensor

sys.path.append("..")
import conv_types as ct
from denoise import model_new

class TestConvConfig(unittest.TestCase):
    def test_simple(self):
        cfg = ct.make_config("k3-s1-128-64-32")
        self.assertEqual(3, len(cfg.layers))
        self.assertEqual([128, 64, 32], [l.out_chan for l in cfg.layers])
    
    def test_channels_down(self):
        cfg = ct.make_config("k3-s1-128-64-32")
        self.assertEqual([3, 128, 64, 32], cfg.get_channels_down(3))
        # self.assertEqual([32, 128, 64, 3], cfg.get_channels_up(32))

    def test_channels_up(self):
        cfg = ct.make_config("k3-s1-128-64-32")
        self.assertEqual([32, 64, 128, 3], cfg.get_channels_up(3))
    
    def test_metadata_dict(self):
        cfg = ct.make_config("k3-s1-128-64-32")
        expected = dict(
            layers_str="k3-s1-128-64-32",
            inner_nonlinearity_type='relu',
            linear_nonlinearity_type='relu',
            final_nonlinearity_type='sigmoid',
            norm_type='layer'
        )
        self.assertEqual(expected, cfg.metadata_dict())

class TestConvConfig_kern3_stride1(unittest.TestCase):
    def test_sizes_down_desired(self):
        cfg = ct.make_config("k3-s1-128-64-32")
        self.assertEqual([128, 128, 128, 128], cfg.get_sizes_down_desired(128))

    def test_sizes_up_desired(self):
        cfg = ct.make_config("k3-s1-128-64-32")
        self.assertEqual([128, 128, 128, 128], cfg.get_sizes_up_desired(128))

    def test_sizes_down_actual(self):
        cfg = ct.make_config("k3-s1-128-64-32")
        self.assertEqual([128, 128, 128, 128], cfg.get_sizes_down_actual(128))

    def test_sizes_up_actual(self):
        cfg = ct.make_config("k3-s1-128-64-32")
        self.assertEqual([128, 128, 128, 128], cfg.get_sizes_up_actual(128))

class TestConvConfig_kern3_stride2(unittest.TestCase):
    def test_sizes_down_desired(self):
        cfg = ct.make_config("k3-s2-128-64-32")
        self.assertEqual([128, 64, 32, 16], cfg.get_sizes_down_desired(128))

    def test_sizes_up_desired(self):
        cfg = ct.make_config("k3-s2-128-64-32")
        self.assertEqual([16, 32, 64, 128], cfg.get_sizes_up_desired(16))

    def test_sizes_down_actual(self):
        cfg = ct.make_config("k3-s2-128-64-32")
        self.assertEqual([128, 64, 32, 16], cfg.get_sizes_down_actual(128))

    def test_sizes_up_actual(self):
        cfg = ct.make_config("k3-s2-128-64-32")
        self.assertEqual([16, 32, 64, 128], cfg.get_sizes_up_desired(16))

class TestConvConfig_kern4_stride1(unittest.TestCase):
    def test_sizes_down_desired(self):
        cfg = ct.make_config("k4-s1-128-64-32")
        self.assertEqual([128, 128, 128, 128], cfg.get_sizes_down_desired(128))

    def test_sizes_up_desired(self):
        cfg = ct.make_config("k4-s1-128-64-32")
        self.assertEqual([128, 128, 128, 128], cfg.get_sizes_up_desired(128))

    def test_sizes_down_actual(self):
        cfg = ct.make_config("k4-s1-128-64-32")
        self.assertEqual([128, 128, 128, 128], cfg.get_sizes_down_actual(128))

    def test_sizes_up_actual(self):
        cfg = ct.make_config("k4-s1-128-64-32")
        self.assertEqual([128, 128, 128, 128], cfg.get_sizes_up_desired(128))

class TestConvConfig_kern4_stride2(unittest.TestCase):
    def test_sizes_down_desired(self):
        cfg = ct.make_config("k4-s2-128-64-32")
        self.assertEqual([128, 64, 32, 16], cfg.get_sizes_down_desired(128))

    def test_sizes_up_desired(self):
        cfg = ct.make_config("k4-s2-128-64-32")
        self.assertEqual([16, 32, 64, 128], cfg.get_sizes_up_desired(16))

    def test_sizes_down_actual(self):
        cfg = ct.make_config("k4-s2-128-64-32")
        self.assertEqual([128, 64, 32, 16], cfg.get_sizes_down_actual(128))

    def test_sizes_up_actual(self):
        cfg = ct.make_config("k4-s2-128-64-32")
        self.assertEqual([16, 32, 64, 128], cfg.get_sizes_up_desired(16))

class TestConvNew(unittest.TestCase):
    def setUp(self):
        self.cfg = ct.make_config("k4-s2-128-64-32")

    def test_init(self):
        size = 256
        net = model_new.VarEncDec(image_size=size, emblen=0, nlinear=0, hidlen=0, cfg=self.cfg)
    
    def test_encode(self):
        size = 256
        emblen = 32
        net = model_new.VarEncDec(image_size=size, emblen=emblen, nlinear=0, hidlen=0, cfg=self.cfg)
        inputs = torch.rand((1, 3, size, size))
        enc_out = net.encode(inputs)
        self.assertEqual(torch.Size([1, emblen]), enc_out.shape)

    def test_decode(self):
        size = 256
        emblen = 32
        net = model_new.VarEncDec(image_size=size, emblen=emblen, nlinear=0, hidlen=0, cfg=self.cfg)
        inputs = torch.rand((1, emblen))
        dec_out = net.decode(inputs)
        self.assertEqual(torch.Size([1, 3, size, size]), dec_out.shape)



import sys
import unittest
from typing import List
from pathlib import Path

import torch
from torch import Tensor

sys.path.append("..")
import conv_types as ct
from denoise import model
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
    def test_init(self):
        cfg = ct.make_config("k4-s2-128-64-32")
        size = 256
        net = model_new.VarEncDec(image_size=size, emblen=0, 
                                  nlinear=0, hidlen=0, cfg=cfg)

class TestDescs(unittest.TestCase):
    def test_encoder_simple(self):
        descs = model.gen_descs("k3-s1-c128,c64,c32")
        self.assertEqual(3, len(descs))
        self.assertEqual([128, 64, 32], [d.channels for d in descs])
        self.assertEqual([3, 3, 3], [d.kernel_size for d in descs])
        self.assertEqual([1, 1, 1], [d.stride for d in descs])

    def test_encoder_invalid_input(self):
        try:
            descs = model.gen_descs("k3-s1-c128-c64-c32")
            self.assertEqual("error expected", "no error")
        except ValueError as e:
            pass

class TestConvEncDec(unittest.TestCase):
    #####
    # kern 3 encoder
    #####
    def test_encoder_kern3_stride1(self):
        size = 64
        net = self.build_net("k3-s1-c32,c64,c128", image_size=size)
        self.assertEqual([128, 64, 64], net.latent_dim)

        input = torch.rand((1, 3, size, size))
        out = net.encoder(input)
        self.check_output_shape([128, 64, 64], out)

    def test_encoder_kern3_stride2(self):
        size = 64
        net = self.build_net("k3-s2-c32,c64,c128", image_size=size)
        self.assertEqual([128, 8, 8], net.latent_dim)

        input = torch.rand((1, 3, size, size))
        out = net.encoder(input)
        self.check_output_shape([128, 8, 8], out)

    def test_decoder_kern3_stride1(self):
        size = 64
        net = self.build_net("k3-s1-c32,c64,c128", image_size=size)

        input = torch.rand((1, *net.latent_dim))
        out = net.decoder(input)
        self.check_output_shape([3, size, size], out)

    def test_decoder_kern3_stride2(self):
        size = 64
        net = self.build_net("k3-s2-c32,c64,c128", image_size=size)

        input = torch.rand((1, *net.latent_dim))
        out = net.decoder(input)
        self.check_output_shape([3, size, size], out)

    #####
    # kern 4 encoder
    #####
    def test_encoder_kern4_stride1(self):
        size = 64
        net = self.build_net("k4-s1-c32,c64,c128", image_size=size)
        self.assertEqual([128, 64, 64], net.latent_dim)

        input = torch.rand((1, 3, size, size))
        out = net.encoder(input)
        self.check_output_shape([128, 64, 64], out)

    def test_encoder_kern4_stride2(self):
        size = 64
        net = self.build_net("k4-s2-c32,c64,c128", image_size=size)
        self.assertEqual([128, 8, 8], net.latent_dim)

        input = torch.rand((1, 3, size, size))
        out = net.encoder(input)
        self.check_output_shape([128, 8, 8], out)

    #####
    # kern 4 decoder
    #####
    def test_decoder_kern4_stride1(self):
        size = 64
        net = self.build_net("k4-s1-c32,c64,c128", image_size=size)

        input = torch.rand((1, *net.latent_dim))
        out = net.decoder(input)
        self.check_output_shape([3, size, size], out)

    def test_decoder_kern4_stride2(self):
        size = 64
        net = self.build_net("k4-s2-c32,c64,c128", image_size=size)

        input = torch.rand((1, *net.latent_dim))
        out = net.decoder(input)
        self.check_output_shape([3, size, size], out)

    def check_output_shape(self, expected: List[int], input: Tensor, remove_batch_dim=True):
        if remove_batch_dim:
            input = input[0]
        input_shape = [s for s in input.shape]
        self.assertEqual(expected, input_shape)
        
    def build_net(self, s: str, 
                  do_variational: bool=False,
                  image_size=128, nchannels=3,
                  emblen=0, nlinear=0, hidlen=0, 
                  do_layernorm=True, do_batchnorm=True, use_bias=True) -> model.ConvEncDec:
        descs = model.gen_descs(s)
        net = model.ConvEncDec(image_size=image_size, nchannels=nchannels, descs=descs,
                               emblen=emblen, nlinear=nlinear, hidlen=hidlen,
                               do_variational=do_variational, do_layernorm=do_layernorm, do_batchnorm=do_batchnorm,
                               use_bias=use_bias)
        return net

    def setUp(self):
        pass


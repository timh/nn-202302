import unittest
from typing import List
from pathlib import Path

import torch
from torch import Tensor

from denoise import model


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
    def test_ldim_kern3_stride1(self):
        net = self.build_net("k3-s1-c32,c64,c128", image_size=64)
        self.assertEqual([128, 64, 64], net.latent_dim)

    def test_ldim_kern3_stride2(self):
        net = self.build_net("k3-s2-c32,c64,c128", image_size=64)
        self.assertEqual([128, 8, 8], net.latent_dim)

    def test_ldim_kern4_stride1(self):
        net = self.build_net("k4-s1-c32,c64,c128", image_size=64)
        self.assertEqual([128, 64, 64], net.latent_dim)

    def test_ldim_kern4_stride2(self):
        net = self.build_net("k4-s2-c32,c64,c128", image_size=64)
        self.assertEqual([128, 8, 8], net.latent_dim)

    def test_encoder_kern3_stride2(self):
        size = 64
        net = self.build_net("k3-s2-c32,c64,c128", image_size=size)

        input = torch.rand((1, 3, size, size))
        out = net.encoder(input)
        self.check_output_shape([128, 8, 8], out)

    def test_decoder_kern3_stride2(self):
        size = 64
        net = self.build_net("k3-s2-c32,c64,c128", image_size=size)

        input = torch.rand((1, *net.latent_dim))
        out = net.decoder(input)
        self.check_output_shape([3, size, size], out)

    def test_encoder_kern4_stride2(self):
        size = 64
        net = self.build_net("k4-s2-c32,c64,c128", image_size=size)

        input = torch.rand((1, 3, size, size))
        out = net.encoder(input)
        self.check_output_shape([128, 8, 8], out)

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


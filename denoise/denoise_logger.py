# %%
import sys
import re
from typing import Deque, List, Dict, Tuple
from pathlib import Path
from collections import deque
from dataclasses import dataclass
import datetime
from PIL import Image
from PIL import ImageDraw, ImageFont
from fonts.ttf import Roboto

import torch
from torch import Tensor
from torchvision import transforms

sys.path.append("..")
import trainer
import loadsave
from denoise_exp import DNExperiment
import model

class DenoiseLogger(trainer.TensorboardLogger):
    save_top_k: int
    top_k_checkpoints: Deque[Path]
    top_k_jsons: Deque[Path]
    top_k_epochs: Deque[int]
    top_k_vloss: Deque[float]
    noise_in: Tensor = None

    def __init__(self, basename: str, truth_is_noise: bool, save_top_k: int, max_epochs: int, num_progress_images: int, device: str):
        super().__init__(f"denoise_{basename}_{max_epochs:04}")

        self.save_top_k = save_top_k
        self.device = device
        self.truth_is_noise = truth_is_noise
        self.num_prog_images = num_progress_images

    def _status_path(self, exp: DNExperiment, subdir: str, epoch: int = 0, suffix = "") -> str:
        filename: List[str] = [exp.label]
        if epoch:
            filename.append(f"epoch_{epoch:04}")
        filename = ",".join(filename)
        path = Path(self.dirname, subdir or "", filename + suffix)
        path.parent.mkdir(exist_ok=True, parents=True)
        return path

    def _ensure_noise(self, image_size: int):
        # TODO: BUG: if we run different experiments with different image_size's, this
        # will break.
        if self.noise_in is None:
            # use the same noise for all experiments & all epochs.
            self.noise_in = model.gen_noise((1, 3, image_size, image_size)).to(self.device) + 0.5

    def _drawtext(self, xy: Tuple[int, int], text: str):
        self._imgdraw.text(xy=xy, text=text, font=self._imgfont, fill="black")
        self._imgdraw.text(xy=(xy[0] + 1, xy[1] + 1), text=text, font=self._imgfont, fill="white")

    def on_exp_start(self, exp: DNExperiment):
        super().on_exp_start(exp)

        self.last_val_loss = None
        self.top_k_checkpoints = deque()
        self.top_k_jsons = deque()
        self.top_k_epochs = deque()
        self.top_k_vloss = deque()
        exp.label += f",nparams_{exp.nparams() / 1e6:.3f}M"

        similar_checkpoints = [(path, exp) for path, exp in loadsave.find_checkpoints()
                               if exp.label == exp.label]
        for ckpt_path, _exp in similar_checkpoints:
            # ckpt_path               = candidate .ckpt file
            # ckpt_path.parent        = "checkpoints" dir
            # ckpt_path.parent.parent = timestamped dir for that run
            status_path = Path(ckpt_path.parent.parent, exp.label + ".status")
            if status_path.exists():
                exp.skip = True
                return
        
        if self.num_prog_images:
            dataset = exp.val_dataloader.dataset
            rand_sampidx = torch.randint(0, len(dataset), (1,))[0]

            input, truth = dataset[rand_sampidx]
            if self.truth_is_noise:
                truth = truth[0]
            else:
                truth = truth[1]
            _chan, width, height = input.shape

            self.image_size = width
            self._margin_x, self._margin_y = 2, 2
            self._margin_noise_y = 20
            self._spacing_y = (self.image_size + self._margin_y)
            self._spacing_x = (self.image_size + self._margin_x)
            self._out_ymin = self._spacing_y * 2                            # 1 input + 1 truth = 2
            self._noise_ymin = self._spacing_y * 3 + self._margin_noise_y   # 1 input + 1 truth + 1 out = 3

            self.prog_path = self._status_path(exp, "images", suffix="-progress.png")
            self.prog_steps = [5, 10, 20, 50]

            img_width = self._spacing_x * self.num_prog_images
            img_height = self._spacing_y * (len(self.prog_steps) + 3) + self._margin_noise_y
            font_size = 14

            net: model.ConvEncDec = exp.net
            # TODO: can't call _imgdraw.textsize() to add to image height because it's not
            # instantiated until the image is.
            if getattr(exp, 'conv_descs', None):
                text = f"{exp.conv_descs}\nnlinear {net.nlinear}, hidlen {net.hidlen}, emblen {net.emblen}\nloss_type {exp.loss_type}, nparams {exp.nparams() / 1e6:.3f}M"
            else:
                text = exp.label
            nlines = len(text.split("\n"))
            text_height = int(nlines * font_size * 1.5)
            img_height += text_height + font_size

            self._img = Image.new("RGB", (img_width, img_height))
            self._imgdraw = ImageDraw.ImageDraw(self._img)
            self._imgfont = ImageFont.truetype(Roboto, font_size)
            xy = (10, img_height - text_height + font_size/2)
            self._imgdraw.text(xy=xy, text=text, font=self._imgfont, fill='white')
            self._img.save(self.prog_path)

            self.prog_every_epochs = exp.max_epochs // self.num_prog_images
            self._imgtransform = transforms.ToPILImage("RGB")
            self._input = input.unsqueeze(0).to(self.device)
            self._input_img = self._imgtransform(input)
            self._truth_img = self._imgtransform(truth)

    def on_exp_end(self, exp: DNExperiment):
        super().on_exp_end(exp)

        if not exp.skip:
            path = Path(self.dirname, f"{exp.label}.status")
            with open(path, "w") as file:
                file.write(str(exp.nepochs))

    def on_epoch_end(self, exp: DNExperiment, epoch: int, train_loss_epoch: float):
        super().on_epoch_end(exp, epoch, train_loss_epoch)

        if self.num_prog_images and (epoch + 1) % self.prog_every_epochs == 0:
            start = datetime.datetime.now()
            idx = epoch // self.prog_every_epochs

            # first two rows are truth, input
            x = idx * self._spacing_x
            y = 0
            self._img.paste(self._truth_img, (x, 0))
            self._drawtext(xy=(x, y), text="truth")

            y = self._spacing_y
            self._img.paste(self._input_img, (x, y))
            self._drawtext(xy=(x, y), text="input")
            self._img.save(self.prog_path)

            # then comes an output.
            y = self._out_ymin

            exp.net.eval()
            out = exp.net(self._input)[0].detach().cpu()
            self._img.paste(self._imgtransform(out), (x, y))
            self._drawtext(xy=(x, y), text=f"epoch {epoch + 1}\noutput")

            # then comes some noise imagination
            self._ensure_noise(self.image_size)
            for i, steps in enumerate(self.prog_steps):
                y = self._noise_ymin + (i * self._spacing_x)
                out = model.generate(exp=exp, num_steps=steps, size=self.image_size, 
                                     truth_is_noise=self.truth_is_noise, 
                                     input=self.noise_in, device=self.device)[0]
                self._img.paste(self._imgtransform(out), (x, y))
                self._drawtext(xy=(x, y), text=f"noise @ {steps} steps")

            self._img.save(self.prog_path)
            end = datetime.datetime.now()
            elapsed = (end - start).total_seconds()
            total_steps = sum(self.prog_steps)
            print(f"  updated progress image in {elapsed:.2f}s ({total_steps} steps)")
    
    def update_val_loss(self, exp: DNExperiment, epoch: int, val_loss: float):
        super().update_val_loss(exp, epoch, val_loss)
        if self.last_val_loss is None or val_loss < self.last_val_loss:
            self.last_val_loss = val_loss

            ckpt_path = self._status_path(exp, "checkpoints", epoch + 1, ".ckpt")
            json_path = self._status_path(exp, "checkpoints", epoch + 1, ".json")

            start = datetime.datetime.now()
            loadsave.save_ckpt_and_metadata(exp, ckpt_path, json_path)
            end = datetime.datetime.now()
            elapsed = (end - start).total_seconds()
            print(f"    saved checkpoint {epoch + 1}: vloss {val_loss:.5f} in {elapsed:.2f}s: {ckpt_path}")

            if self.save_top_k > 0:
                self.top_k_checkpoints.append(ckpt_path)
                self.top_k_jsons.append(json_path)
                self.top_k_epochs.append(epoch)
                self.top_k_vloss.append(val_loss)
                if len(self.top_k_checkpoints) > self.save_top_k:
                    to_remove_ckpt = self.top_k_checkpoints.popleft()
                    to_remove_json = self.top_k_jsons.popleft()
                    removed_epoch = self.top_k_epochs.popleft()
                    removed_vloss = self.top_k_vloss.popleft()
                    to_remove_ckpt.unlink()
                    to_remove_json.unlink()
                    print(f"  removed checkpoint {removed_epoch + 1}: vloss {removed_vloss:.5f}")

def in_notebook():
    # https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


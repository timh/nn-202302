# %%
import sys
import re
from typing import Deque, List, Dict, Tuple
from pathlib import Path
from collections import deque
from dataclasses import dataclass
import datetime
import matplotlib.pyplot as plt
from IPython import display
from PIL import Image
from PIL import ImageDraw, ImageFont
from fonts.ttf import Roboto

import torch
from torch import Tensor
from torchvision import transforms

sys.path.append("..")
import trainer
import experiment
from experiment import Experiment
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

        nrows = 3
        ncols = 5
        base_dim = 6
        plt.gcf().set_figwidth(base_dim * ncols)
        plt.gcf().set_figheight(base_dim * nrows)

        out_title = "output (noise)" if truth_is_noise else "output (denoised src)"
        noise_title = "truth (noise)" if truth_is_noise else "added noise"

        self.axes_in_noised = plt.subplot(nrows, ncols, 1, title="input (src + noise)")
        self.axes_out = plt.subplot(nrows, ncols, 2, title=out_title)
        self.axes_truth_noise = plt.subplot(nrows, ncols, 3, title=noise_title)
        if truth_is_noise:
            self.axes_in_sub_out = plt.subplot(nrows, ncols, 4, title="in - out (input w/o noise)")
        self.axes_src = plt.subplot(nrows, ncols, 5, title="truth (src)")

        self.axes_gen = {val: plt.subplot(nrows, ncols, 6 + i, title=f"{val} steps") 
                         for i, val in enumerate([1, 2, 3, 4, 5, 10, 20, 30, 40, 50])}
        
        self.save_top_k = save_top_k
        self.device = device
        self.truth_is_noise = truth_is_noise

        self.num_prog_images = num_progress_images

    def _status_path(self, exp: Experiment, subdir: str, epoch: int = 0, suffix = "") -> str:
        filename: List[str] = [exp.label]
        if epoch:
            filename.append(f"epoch_{epoch + 1:04}")
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
        self.prog_draw.text(xy=xy, text=text, font=self.prog_font, fill="black")
        self.prog_draw.text(xy=(xy[0] + 1, xy[1] + 1), text=text, font=self.prog_font, fill="white")

    def on_exp_start(self, exp: Experiment):
        super().on_exp_start(exp)

        self.last_val_loss = None
        self.top_k_checkpoints = deque()
        self.top_k_jsons = deque()
        self.top_k_epochs = deque()
        self.top_k_vloss = deque()
        exp.label += f",nparams_{exp.nparams() / 1e6:.3f}M"

        similar_checkpoints = [(path, exp) for path, exp in find_all_checkpoints(Path("runs"))
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

            rand_input, rand_truth = dataset[rand_sampidx]
            if self.truth_is_noise:
                rand_truth = rand_truth[0]
            else:
                rand_truth = rand_truth[1]
            _chan, width, height = rand_input.shape

            self.image_size = width
            self._margin_epoch_y = 2
            self._margin_noise_x = self.image_size // 2

            self.prog_path = self._status_path(exp, "images", suffix="-progress.png")
            self.prog_steps = [20, 50]
            self.prog_width = width * len(self.prog_steps) + self._margin_noise_x
            self.prog_height = (height + self._margin_epoch_y) * (self.num_prog_images + 2)
            self.prog_img = Image.new("RGB", (self.prog_width, self.prog_height))

            self.prog_draw = ImageDraw.ImageDraw(self.prog_img)
            self.prog_font = ImageFont.truetype(Roboto, 14)

            self.prog_every_epochs = exp.max_epochs // self.num_prog_images
            self.prog_transform = transforms.ToPILImage("RGB")
            self.prog_input = rand_input.unsqueeze(0).to(self.device)

            self.prog_img.paste(self.prog_transform(rand_truth), (0, 0))
            self._drawtext(xy=(0, 0), text="truth")

            y = height + self._margin_epoch_y
            self.prog_img.paste(self.prog_transform(rand_input), (0, y))
            self._drawtext(xy=(0, y), text="input")
            self.prog_img.save(self.prog_path)


    def on_exp_end(self, exp: Experiment):
        super().on_exp_end(exp)

        if not exp.skip:
            path = Path(self.dirname, f"{exp.label}.status")
            with open(path, "w") as file:
                file.write(str(exp.nepochs))

    def on_epoch_end(self, exp: Experiment, epoch: int, train_loss_epoch: float):
        super().on_epoch_end(exp, epoch, train_loss_epoch)

        if self.num_prog_images and epoch % self.prog_every_epochs == 0:
            idx = epoch // self.prog_every_epochs
            # first two rows are truth, input
            y = (idx + 2) * (self.image_size + self._margin_epoch_y)

            self._ensure_noise(self.image_size)

            exp.net.eval()
            out = exp.net(self.prog_input)[0].detach().cpu()
            self.prog_img.paste(self.prog_transform(out), (0, y))
            self._drawtext(xy=(0, y), text=f"epoch {epoch + 1}")

            for i, steps in enumerate(self.prog_steps):
                x = (i + 1) * self.image_size + self._margin_noise_x
                out = model.generate(exp=exp, num_steps=steps, size=self.image_size, truth_is_noise=self.truth_is_noise, input=self.noise_in, device=self.device)
                out = exp.net(self.noise_in)[0]
                self.prog_img.paste(self.prog_transform(out), (x, y))
                self._drawtext(xy=(x, y), text=f"noise -> {steps} steps")

            self.prog_img.save(self.prog_path)
            print(f"  updated progress image")
    
    def update_val_loss(self, exp: Experiment, epoch: int, val_loss: float):
        super().update_val_loss(exp, epoch, val_loss)
        if self.last_val_loss is None or val_loss < self.last_val_loss:
            self.last_val_loss = val_loss

            ckpt_path = self._status_path(exp, "checkpoints", epoch, ".ckpt")
            json_path = self._status_path(exp, "checkpoints", epoch, ".json")

            start = datetime.datetime.now()
            experiment.save_ckpt_and_metadata(exp, ckpt_path, json_path)
            end = datetime.datetime.now()
            elapsed = (end - start).total_seconds()
            print(f"    saved checkpoint {epoch + 1}: vloss {val_loss:.5f} in {elapsed:.2f}s")

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

    def print_status(self, exp: Experiment, epoch: int, batch: int, batches: int, train_loss: float):
        super().print_status(exp, epoch, batch, batches, train_loss)

        in_noised = exp.last_train_in[-1]
        out_noise = exp.last_train_out[-1]
        truth_noise = exp.last_train_truth[-1][0]
        src = exp.last_train_truth[-1][1]

        chan, width, height = in_noised.shape

        def transpose(img: Tensor) -> Tensor:
            img = img.clamp(min=0, max=1)
            img = img.detach().cpu()
            return torch.permute(img, (1, 2, 0))

        self.axes_in_noised.imshow(transpose(in_noised))
        self.axes_out.imshow(transpose(out_noise))
        self.axes_truth_noise.imshow(transpose(truth_noise))
        if self.truth_is_noise:
            self.axes_in_sub_out.imshow(transpose(in_noised - out_noise))
        self.axes_src.imshow(transpose(src))

        self._ensure_noise(width)

        for i, (val, axes) in enumerate(self.axes_gen.items()):
            gen = model.generate(exp, val, width, truth_is_noise=self.truth_is_noise, input=self.noise_in, device=self.device)[0]
            self.axes_gen[val].imshow(transpose(gen))

        if in_notebook():
            display.display(plt.gcf())
        img_path = self._status_path(exp, "images", epoch, ".png")
        plt.savefig(img_path)
        print(f"  saved PNG to {img_path}")
        print()


# conv_encdec2_k3-s2-op1-p1-c32,c64,c64,emblen_384,nlin_1,hidlen_128,bnorm,slr_1.0E-03,batch_128,cnt_2,nparams_12.860M,epoch_0739,vloss_0.10699.ckpt
def find_all_checkpoints(runsdir: Path) -> List[Tuple[Path, Experiment]]:
    res: List[Tuple[Path, Experiment]] = list()
    for run_path in runsdir.iterdir():
        if not run_path.is_dir():
            continue
        checkpoints = Path(run_path, "checkpoints")
        if not checkpoints.exists():
            continue

        for ckpt_path in checkpoints.iterdir():
            if not ckpt_path.name.endswith(".ckpt"):
                continue
            meta_path = Path(str(ckpt_path)[:-5] + ".json")
            exp = experiment.load_experiment_metadata(meta_path)
            res.append((ckpt_path, exp))

    return res

if __name__ == "__main__":
    all_cp = find_all_checkpoints()
    print("all:")
    print("\n".join(map(str, all_cp)))
    print()

    label = "k3-s2-op1-p1-c32,c64,c64,emblen_384,nlin_3,hidlen_128,bnorm,slr_1.0E-03,batch_128,cnt_2,nparams_12.828M"
    filter_cp = [cp for cp in all_cp if cp.label == label]
    print("matching:")
    print("\n".join(map(str, filter_cp)))
    print()
            
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


# %%
import datetime
import sys
import math
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple, Union, Callable
from PIL import Image, ImageDraw, ImageFont
from fonts.ttf import Roboto


from torch import Tensor
from torchvision import transforms

sys.path.append("..")
import trainer
from experiment import Experiment
import image_util

class ImageProgressGenerator:
    def on_exp_start(self, exp: Experiment, nrows: int):
        pass

    def get_exp_descrs(self, exps: List[Experiment]) -> List[Union[str, List[str]]]:
        descrs = [exp.describe() for exp in exps]
        return descrs

    """
    returns headers of columns that should be in the image, to the left
    of the columns for each experiment.
    """
    def get_col_headers(self) -> List[str]:
        pass

    """
    returns 3D tensor (chan, width, height) for the header images, if any, that
    sould go in the given row.
    """
    def get_col_header_images(self, exp: Experiment, row: int) -> List[Tensor]:
        pass

    """
    returns 3D tensor (chan, width, height) for the image that should
    go in the given row.
    """
    def get_image(self, exp: Experiment, row: int) -> Tensor:
        pass

"""
Generate a unique image for each Experiment.
It has an arbitrary number of rows and columns.

If padding around images is desired, pass in a larger image_size.
"""
class ImageProgressLogger(trainer.TrainerLogger):
    progress_every_nepochs: int = 0

    _content_x = None
    _content_y = None

    _padding = 2
    _path: Path
    _image: Image.Image
    _draw: ImageDraw.ImageDraw
    _font: ImageFont.ImageFont
    _to_image: transforms.ToPILImage

    image_size: int

    exps: List[Experiment]
    nrows: int
    col_headers: List[str]
    _exp_descr_xy: Tuple[int, int]
    _last_row_header_drawn: int

    def __init__(self,
                 dirname: str,
                 progress_every_nepochs: int,
                 image_size: int,
                 generator: ImageProgressGenerator,
                 exps: List[Experiment]):
        super().__init__(dirname)
        self.progress_every_nepochs = progress_every_nepochs
        self.image_size = image_size
        self.generator = generator
        self._to_image = transforms.ToPILImage("RGB")
        self.exps = exps

        self._path = None
        self._image = None

        min_epoch = min([exp.nepochs for exp in exps])
        max_epoch = min([exp.max_epochs for exp in exps])
        self.nrows = (max_epoch - min_epoch) // self.progress_every_nepochs
        self.nrows = max(1, self.nrows)
        self.col_headers = generator.get_col_headers()
    
    def _pos_for(self, *, row: int, col: int) -> Tuple[int, int]:
        x = self._content_x + col * self.image_size
        y = self._content_y + row * self.image_size
        return x, y
    
    def _create_image(self):
        # heuristic
        font_size = max(10, math.ceil(self.image_size / 15))
        self._font = ImageFont.truetype(Roboto, font_size)

        exp_descrs = self.generator.get_exp_descrs(self.exps)
        exp_descrs_fit, descr_max_height = \
            image_util.fit_strings_multi(exp_descrs,
                                         max_width=self.image_size,
                                         font=self._font)

        # BUG: this will probably fail if image_size isn't the same across 
        # all the experiments.
        col_epoch_width = self.image_size // 2
        self._col_headers_x = col_epoch_width
        self._content_x = col_epoch_width + len(self.col_headers) * self.image_size
        self._content_y = descr_max_height

        width = self._content_x + len(self.exps) * self.image_size
        height = self._content_y + self.nrows * self.image_size

        self._image = Image.new("RGB", (width, height))
        self._draw = ImageDraw.ImageDraw(self._image)
        self._path = Path(self.dirname, "run-progress.png")

        # draw (extra) column headers, if any
        for col, col_header in enumerate(self.col_headers):
            header_x = self._col_headers_x + col * self.image_size
            self._draw.text(xy=(header_x, 0), text=col_header, font=self._font, fill='white')

    def _draw_exp_descr(self, exp: Experiment):
        exp_descr = self.generator.get_exp_descrs([exp])[0]
        exp_descr_fit, _max = \
            image_util.fit_strings(exp_descr,
                                   max_width=self.image_size,
                                   font=self._font)

        left, top = self._exp_descr_xy
        right, bot = left + self.image_size, self._content_y
        box = (left, top, right, bot)

        self._draw.rectangle(xy=box, fill='black')
        self._draw.text(xy=self._exp_descr_xy, text=exp_descr_fit,
                        font=self._font, fill='white')
    
    def _current_row(self, exp: Experiment, epoch: int) -> int:
        nepoch = epoch - exp.start_epoch()
        row = nepoch // self.progress_every_nepochs
        return row

    def on_exp_start(self, exp: Experiment):
        if self._image is None:
            self._create_image()

        # draw the experiment description at the start
        exp_descr_x, _y = self._pos_for(row=0, col=exp.exp_idx)
        self._exp_descr_xy = (exp_descr_x, 0)
        self._draw_exp_descr(exp)
        self._last_row_header_drawn = -1

        # print(f"{exp.max_epochs=} {exp.start_epoch()=} {self.progress_every_nepochs=} {self.nrows=}")

        self.generator.on_exp_start(exp, self.nrows)
        self._image.save(self._path)

    # NOTE: should I comment it out to avoid CUDA OOM?
    def print_status(self, exp: Experiment, epoch: int, 
                     _batch: int, _exp_batch: int, 
                     train_loss_epoch: float):
        self.update_image(exp=exp, epoch=epoch, train_loss_epoch=train_loss_epoch)
    
    def on_epoch_end(self, exp: Experiment, epoch: int, train_loss_epoch: float):
        if (epoch + 1) % self.progress_every_nepochs == 0:
            row = self._current_row(exp, epoch)

            self.update_image(exp=exp, epoch=epoch, train_loss_epoch=train_loss_epoch)

    def update_image(self, exp: Experiment, epoch: int, train_loss_epoch: float):
        start = datetime.datetime.now()

        row = self._current_row(exp, epoch)
        self._draw_exp_descr(exp)

        # update epoch label
        _row_x, row_y = self._pos_for(row=row, col=0)
        row_label = [f"epoch {epoch + 1}",
                     f"tl {train_loss_epoch:.3f}",
                     f"vl {exp.lastepoch_val_loss:.3f}"]
        row_label = "\n".join(row_label)
        self._draw.rectangle(xy=(0, row_y, self._col_headers_x, row_y + self.image_size), fill='black')
        self._draw.text(xy=(0, row_y), text=row_label, font=self._font, fill='white')

        # update header images, if any
        if row != self._last_row_header_drawn:
            header_images_t = self.generator.get_col_header_images(row)
            for header_img_idx, header_image_t in enumerate(header_images_t):
                x = self._col_headers_x + header_img_idx * self.image_size
                y = self._content_y + row * self.image_size
                header_image = self._to_image(header_image_t)
                self._image.paste(header_image, box=(x, y))
            self._last_row_header_drawn = row

        image_t = self.generator.get_image(exp, row)
        image = self._to_image(image_t)

        xy = self._pos_for(row=row, col=exp.exp_idx)
        self._image.paste(image, box=xy)
        
        self._image.save(self._path)
        symlink_path = Path("runs", "last-run-progress.png")
        symlink_path.unlink(missing_ok=True)
        symlink_path.symlink_to(self._path.absolute())

        end = datetime.datetime.now()
        print(f"  updated progress image in {(end - start).total_seconds():.3f}s")



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
        return []

    """
    returns 3D tensor (chan, width, height) for the header images, if any, that
    sould go in the given row.
    """
    def get_col_header_images(self, row: int) -> List[Tensor]:
        return []

    """
    get number of images that will be rendered per experiment.
    """
    def get_num_images_per_exp(self) -> int:
        return 1

    """
    get image labels for the images returned by get_images. these will
    be combined with the experiment description.
    """    
    def get_image_labels(self) -> List[str]:
        return []

    """
    returns 3D tensor (chan, width, height) for the image that should
    go in the given row.
    """
    def get_images(self, exp: Experiment, row: int) -> List[Tensor]:
        raise NotImplementedError("override this")

"""
Generate a unique image for each Experiment.
It has an arbitrary number of rows and columns.
"""
class ImageProgressLogger(trainer.TrainerLogger):
    progress_every_nepochs: int = 0

    content_x = None
    content_y = None

    path: Path
    image: Image.Image
    draw: ImageDraw.ImageDraw
    font: ImageFont.ImageFont
    _to_image: transforms.ToPILImage

    image_size: int

    exps: List[Experiment]
    nrows: int
    leftcol_headers: List[str]
    image_labels: List[str]

    exp_descr_xy: Tuple[int, int]

    last_row_header_drawn: int

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

        self.path = None
        self.image = None

        self._min_epochs = min([exp.nepochs for exp in exps])
        self._max_epochs = min([exp.max_epochs for exp in exps])
        self.nrows = (self._max_epochs - self._min_epochs) // self.progress_every_nepochs
        self.nrows = max(1, self.nrows)

        # TODO: rename leftcol_headers to something else.
        self.leftcol_headers = generator.get_col_headers()
    
    def _pos_for(self, *, row: int, col: int) -> Tuple[int, int]:
        x = self.content_x + col * self.image_size
        y = self.content_y + row * self.image_size
        return x, y
    
    def _create_image(self):
        # heuristic
        self.nimage_per_exp = self.generator.get_num_images_per_exp()
        self.image_labels = [[label] for label in self.generator.get_image_labels()]
        self.exp_width = self.image_size * self.nimage_per_exp

        font_size = max(10, math.ceil(self.image_size / 15))
        self.font = ImageFont.truetype(Roboto, font_size)

        exp_descrs = self.generator.get_exp_descrs(self.exps)
        exp_descrs_fit, descr_max_height = \
            image_util.fit_strings_multi(exp_descrs,
                                         max_width=self.exp_width,
                                         font=self.font)
        if len(self.image_labels) > 0:
            image_labels_fit, labels_max_height = \
                image_util.fit_strings_multi(self.image_labels,
                                             max_width=self.exp_width,
                                             font=self.font)
            self.image_labels = image_labels_fit
        else:
            self.image_labels = []
            labels_max_height = 0

        self.image_labels_y = descr_max_height


        # BUG: this will probably fail if image_size isn't the same across 
        # all the experiments.
        col_epoch_width = self.image_size // 2
        self.col_headers_x = col_epoch_width
        self.content_x = col_epoch_width + len(self.leftcol_headers) * self.image_size
        self.content_y = descr_max_height + labels_max_height

        width = (self.content_x + 
                 len(self.exps) * self.image_size * self.nimage_per_exp)
        height = self.content_y + self.nrows * self.image_size
        print(f"{height=} {self.content_y=} {self.nrows=} {self.image_size=}")

        self.image = Image.new("RGB", (width, height))
        self.draw = ImageDraw.ImageDraw(self.image)
        self.path = Path(self.dirname, "run-progress.png")

        # draw (extra) column headers, if any
        for col, col_header in enumerate(self.leftcol_headers):
            header_x = self.col_headers_x + col * self.image_size
            self.draw.text(xy=(header_x, 0), text=col_header, font=self.font, fill='white')

    def _draw_exp_header(self, exp: Experiment):
        exp_descr = self.generator.get_exp_descrs([exp])[0]
        exp_descr_fit, _max = \
            image_util.fit_strings(exp_descr,
                                   max_width=self.exp_width,
                                   font=self.font)

        left, top = self.exp_descr_xy
        right, bot = left + self.exp_width, self.content_y
        box = (left, top, right, bot)

        self.draw.rectangle(xy=box, fill='black')
        self.draw.text(xy=self.exp_descr_xy, text=exp_descr_fit,
                        font=self.font, fill='white')

        for label_idx, label in enumerate(self.image_labels):
            col = (exp.exp_idx * len(self.image_labels)) + label_idx
            label_x = self.content_x + col * self.image_size
            label_y = self.image_labels_y
            self.draw.text(xy=(label_x, label_y), text=label,
                           font=self.font, fill='green')
    
    def _current_row(self, exp: Experiment, epoch: int) -> int:
        nepoch = epoch - self._min_epochs
        row = nepoch // self.progress_every_nepochs
        return row

    def on_exp_start(self, exp: Experiment):
        if self.image is None:
            self._create_image()

        # draw the experiment description at the start
        exp_descr_x, _y = self._pos_for(row=0, col=exp.exp_idx)
        self.exp_descr_xy = (exp_descr_x, 0)
        self._draw_exp_header(exp)
        self.last_row_header_drawn = -1

        # print(f"{exp.max_epochs=} {exp.start_epoch()=} {self.progress_every_nepochs=} {self.nrows=}")

        self.generator.on_exp_start(exp, self.nrows)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(self.path)

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
        self._draw_exp_header(exp)

        # update epoch label
        _row_x, row_y = self._pos_for(row=row, col=0)
        row_label = [f"epoch {epoch + 1}",
                     f"tl {train_loss_epoch:.3f}",
                     f"vl {exp.lastepoch_val_loss:.3f}"]
        row_label = "\n".join(row_label)
        self.draw.rectangle(xy=(0, row_y, self.col_headers_x, row_y + self.image_size), fill='black')
        self.draw.text(xy=(0, row_y), text=row_label, font=self.font, fill='white')

        # update header images, if any
        if row != self.last_row_header_drawn:
            header_images_t = self.generator.get_col_header_images(row)
            for header_img_idx, header_image_t in enumerate(header_images_t):
                x = self.col_headers_x + header_img_idx * self.image_size
                y = self.content_y + row * self.image_size
                header_image = self._to_image(header_image_t)
                self.image.paste(header_image, box=(x, y))
            self.last_row_header_drawn = row

        images_t = self.generator.get_images(exp, row)
        images = [self._to_image(image_t) for image_t in images_t]

        for img_idx, image in enumerate(images):
            col = self.nimage_per_exp * exp.exp_idx + img_idx
            xy = self._pos_for(row=row, col=col)
            self.image.paste(image, box=xy)
        
        self.image.save(self.path)
        symlink_path = Path("runs", "last-run-progress.png")
        symlink_path.unlink(missing_ok=True)
        symlink_path.symlink_to(self.path.absolute())

        end = datetime.datetime.now()
        print(f"  updated progress image in {(end - start).total_seconds():.3f}s")



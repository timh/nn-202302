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

COLOR_COL_LABEL = "green"

class ImageProgressGenerator:
    def on_exp_start(self, exp: Experiment, nrows: int):
        pass

    def get_exp_descrs(self, exps: List[Experiment]) -> List[Union[str, List[str]]]:
        descrs = [exp.describe() for exp in exps]
        return descrs

    """
    returns labels of columns that should be in the image, to the left
    of the columns for each experiment.
    """
    def get_fixed_labels(self) -> List[str]:
        return []

    """
    returns 3D tensor (chan, width, height) for the header images, if any, that
    sould go in the given row.
    """
    def get_fixed_images(self, row: int) -> List[Tensor]:
        return []

    """
    get number of images that will be rendered per experiment.
    """
    def get_exp_num_cols(self) -> int:
        return 1

    """
    get image labels for the images returned by get_exp_images. these will
    be combined with the experiment description.
    """    
    def get_exp_col_labels(self) -> List[str]:
        return []

    """
    returns 3D tensor (chan, width, height) for the image that should
    go in the given row.
    """
    def get_exp_images(self, exp: Experiment, row: int) -> List[Tensor]:
        raise NotImplementedError("override this")

"""
Generate a unique image for each Experiment.
It has an arbitrary number of rows and columns.
"""
class ImageProgressLogger(trainer.TrainerLogger):
    progress_every_nepochs: int = 0

    path: Path
    image: Image.Image
    draw: ImageDraw.ImageDraw
    font: ImageFont.ImageFont
    _to_image: transforms.ToPILImage

    image_size: int

    exps: List[Experiment]
    nrows: int

    fixed_labels: List[str]       # fixed on the left on the image. one set only.
    exp_labels: List[str]         # labels for columns rendered for each experiment.
    exp_ncols: int

    content_x: int                # where experiment content starts
    content_y: int
    fixed_labels_x: int           # where fixed labels start
    last_row_fixed_drawn: int     # last row where we drew the fixed images

    exp_descr_xy: Tuple[int, int]

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

    def _create_image(self):
        self.fixed_labels = self.generator.get_fixed_labels()
        self.exp_labels = [[label] for label in self.generator.get_exp_col_labels()]
        self.exp_ncols = self.generator.get_exp_num_cols()
    
        self.exp_width = self.image_size * self.exp_ncols

        font_size = max(10, math.ceil(self.image_size / 15))
        self.font = ImageFont.truetype(Roboto, font_size)

        # experiment descriptions
        exp_descrs = self.generator.get_exp_descrs(self.exps)
        exp_descrs_fit, descr_max_height = \
            image_util.fit_strings_multi(exp_descrs,
                                         max_width=self.exp_width,
                                         font=self.font)

        # experiment column labels
        if len(self.exp_labels) > 0:
            exp_labels_fit, exp_labels_height = \
                image_util.fit_strings_multi(self.exp_labels,
                                             max_width=self.exp_width,
                                             font=self.font)
            self.exp_labels = exp_labels_fit
        else:
            self.exp_labels = []
            exp_labels_height = 0

        # all labels (fixed and experiment) go at the bottom of the
        # header row.
        self.labels_y = descr_max_height

        # BUG: this will probably fail if image_size isn't the same across 
        # all the experiments.
        col_epoch_width = self.image_size // 2
        self.fixed_labels_x = col_epoch_width
        self.content_x = col_epoch_width + len(self.fixed_labels) * self.image_size
        self.content_y = descr_max_height + exp_labels_height

        width = (self.content_x + 
                 len(self.exps) * self.image_size * self.exp_ncols)
        height = self.content_y + self.nrows * self.image_size

        self.image = Image.new("RGB", (width, height))
        self.draw = ImageDraw.ImageDraw(self.image)

        # draw fixed labels, if any
        for col, fixed_label in enumerate(self.fixed_labels):
            label_x = self.fixed_labels_x + col * self.image_size
            self.draw.text(xy=(label_x, self.labels_y), font=self.font, 
                           text=fixed_label, fill=COLOR_COL_LABEL)

        # setup image paths and make a symlink
        self.path = Path(self.dirname, "run-progress.png")
        self.path_temp = Path(str(self.path).replace(".png", "-tmp.png"))
        symlink_path = Path("runs", "last-run-progress.png")
        symlink_path.unlink(missing_ok=True)
        symlink_path.symlink_to(self.path.absolute())

    def _pos_for(self, *, row: int, col: int) -> Tuple[int, int]:
        x = self.content_x + col * self.image_size
        y = self.content_y + row * self.image_size
        return x, y
    
    def _draw_exp_column_labels(self, exp: Experiment):
        # experiment description on top.
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

        # then experiment labels.
        for label_idx, label in enumerate(self.exp_labels):
            col = (exp.exp_idx * len(self.exp_labels)) + label_idx
            label_x = self.content_x + col * self.image_size
            label_y = self.labels_y
            self.draw.text(xy=(label_x, label_y), text=label,
                           font=self.font, fill=COLOR_COL_LABEL)
    
    def _current_row(self, exp: Experiment, epoch: int) -> int:
        nepoch = epoch - self._min_epochs
        row = nepoch // self.progress_every_nepochs
        return row

    def on_exp_start(self, exp: Experiment):
        if self.image is None:
            self._create_image()

        # draw the experiment description at the start
        exp_descr_x, _y = self._pos_for(row=0, col=exp.exp_idx * self.exp_ncols)
        self.exp_descr_xy = (exp_descr_x, 0)
        self._draw_exp_column_labels(exp)
        self.last_row_fixed_drawn = -1

        # print(f"{exp.max_epochs=} {exp.start_epoch()=} {self.progress_every_nepochs=} {self.nrows=}")

        self.generator.on_exp_start(exp, self.nrows)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(self.path)

    def print_status(self, exp: Experiment, epoch: int, 
                     _batch: int, _exp_batch: int, 
                     train_loss_epoch: float):
        self.update_image(exp=exp, epoch=epoch, train_loss_epoch=train_loss_epoch)
    
    def on_epoch_end(self, exp: Experiment, epoch: int, train_loss_epoch: float):
        if ((epoch + 1) % self.progress_every_nepochs == 0 or
            epoch == exp.max_epochs - 1):
            row = self._current_row(exp, epoch)

            self.update_image(exp=exp, epoch=epoch, train_loss_epoch=train_loss_epoch)

    def update_image(self, exp: Experiment, epoch: int, train_loss_epoch: float):
        start = datetime.datetime.now()

        row = self._current_row(exp, epoch)
        self._draw_exp_column_labels(exp)

        # update epoch label
        _row_x, row_y = self._pos_for(row=row, col=0)
        row_label = [f"epoch {epoch + 1}",
                     f"tl {train_loss_epoch:.3f}",
                     f"vl {exp.lastepoch_val_loss:.3f}"]
        row_label = "\n".join(row_label)
        self.draw.rectangle(xy=(0, row_y, self.fixed_labels_x, row_y + self.image_size), fill='black')
        self.draw.text(xy=(0, row_y), text=row_label, font=self.font, fill='white')

        # update fixed column images, if any
        if row != self.last_row_fixed_drawn:
            header_images_t = self.generator.get_fixed_images(row)
            for header_img_idx, header_image_t in enumerate(header_images_t):
                x = self.fixed_labels_x + header_img_idx * self.image_size
                y = self.content_y + row * self.image_size
                header_image = self._to_image(header_image_t)
                self.image.paste(header_image, box=(x, y))
            self.last_row_fixed_drawn = row

        images_t = self.generator.get_exp_images(exp, row)
        images = [self._to_image(image_t) for image_t in images_t]

        for img_idx, image in enumerate(images):
            col = self.exp_ncols * exp.exp_idx + img_idx
            xy = self._pos_for(row=row, col=col)
            self.image.paste(image, box=xy)

        # save image to temp path, rename to real path
        self.image.save(self.path_temp)
        self.path_temp.rename(self.path)

        end = datetime.datetime.now()
        print(f"  updated progress image in {(end - start).total_seconds():.3f}s")



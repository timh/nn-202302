import datetime
import sys
import math
from pathlib import Path
from typing import List, Tuple, Union, Callable
from PIL import Image, ImageDraw, ImageFont
from fonts.ttf import Roboto

import torch
from torch import Tensor

sys.path.append("..")
from nnexp.training import trainer
from nnexp.experiment import Experiment
from nnexp.images import image_util

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
    Returns 3D tensor (chan, width, height) for the header images, if any, that
    sould go in the given row. Optionally return an annotation for each image, which
    will berendered on top of it.
    """
    def get_fixed_images(self, row: int) -> List[Union[Tuple[Tensor, str], Tensor]]:
        return []

    """
    Get number of images/annotations that will be rendered per experiment.
    """
    def get_exp_num_cols(self) -> int:
        return 1

    """
    Get header row labels for the images returned by get_exp_images. these will
    be combined with the experiment description in the header.
    """    
    def get_exp_col_labels(self) -> List[str]:
        return []

    """
    returns 3D tensor (chan, width, height) for the image that should
    go in the given row.
    """
    def get_exp_images(self, exp: Experiment, row: int, train_loss_epoch: float) -> List[Union[Tuple[Tensor, str], Tensor]]:
        raise NotImplementedError("override this")

"""
Generate a unique image for each Experiment.
It has an arbitrary number of rows and columns.
"""
# TODO: add "do_group_columns" option. all of the 0th exp_images are adjacent, 
# then the [1]th, etc.
# NOTE: exp labels won't fill as well.
#
# so, instead of
#     | exp[0].images[0] | exp[0].images[1] | exp[0].images[2]  |  exp[1].images[0] | exp[1].images[1] | exp[1].images[2]
#
# do:
#     | exp[0].images[0] | exp[1].images[0]  |  exp[0].images[1] | exp1.images[1]  |  exp[0].images[2] | exp[1].images[2]


class ImageProgressLogger(trainer.TrainerLogger):
    progress_every_nepochs: int = 0

    path: Path
    image: Image.Image
    draw: ImageDraw.ImageDraw
    font: ImageFont.ImageFont

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
                 *,
                 basename: str, started_at: datetime.datetime = None,
                 progress_every_nepochs: int,
                 image_size: int,
                 generator: ImageProgressGenerator,
                 exps: List[Experiment]):
        super().__init__(basename, started_at=started_at)
        self.progress_every_nepochs = progress_every_nepochs
        self.image_size = image_size
        self.generator = generator
        self.exps = exps

        self.path = None
        self.image = None

        self._min_epochs = min([exp.nepochs for exp in exps])
        self._max_epochs = min([exp.max_epochs for exp in exps])
        self.nrows = (self._max_epochs - self._min_epochs) // self.progress_every_nepochs
        self.nrows = max(1, self.nrows)

        exp_shortcodes = sorted([exp.shortcode for exp in exps])
        exp_shortcodes = ",".join(exp_shortcodes)
        if len(exp_shortcodes) > 27:
            # shortcodes, no trailing comma
            exp_shortcodes = exp_shortcodes[:27] + "..."
        run_dir = self.get_run_dir("images", include_timestamp=False)

        self.path = Path(run_dir, f"run-progress--{exp_shortcodes}--{self.started_at_str}.png")
        self.path_temp = Path(str(self.path).replace(".png", "-tmp.png"))

    def _create_image(self):
        self.fixed_labels = self.generator.get_fixed_labels()
        self.exp_labels = [[label] for label in self.generator.get_exp_col_labels()]
        self.exp_ncols = self.generator.get_exp_num_cols()
    
        self.exp_width = self.image_size * self.exp_ncols

        font_size = max(8, math.ceil(self.image_size / 15))
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
        self.draw.line((0, descr_max_height, width, descr_max_height), fill="red")
        self.draw.line((0, descr_max_height + exp_labels_height, width, descr_max_height + exp_labels_height), fill="yellow")

        # draw fixed labels, if any
        for col, fixed_label in enumerate(self.fixed_labels):
            label_x = self.fixed_labels_x + col * self.image_size
            self.draw.text(xy=(label_x, self.labels_y), font=self.font, 
                           text=fixed_label, fill=COLOR_COL_LABEL)

        # setup image paths and make a symlink
        symlink_path = Path(image_util.DEFAULT_DIR, "last-run-progress.png")
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
        row = min(row, self.nrows - 1)
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

    def print_status(self, exp: Experiment, 
                     _batch: int, _exp_batch: int, 
                     train_loss_epoch: float):
        self.update(exp=exp, train_loss_epoch=train_loss_epoch)
    
    def on_epoch_end(self, exp: Experiment):
        if ((exp.nepochs + 1) % self.progress_every_nepochs == 0 or
            exp.nepochs == exp.max_epochs - 1):
            row = self._current_row(exp, exp.nepochs)

            self.update(exp=exp, train_loss_epoch=exp.last_train_loss)

    def _draw_image_tuple(self, *, xy: Tuple[int, int], image_tuple: Union[Tuple[Tensor, str], Tensor]):
        if type(image_tuple) in [list, tuple]:
            image_t, anno = image_tuple
        else:
            image_t = image_tuple
            anno = None

        image = image_util.tensor_to_pil(image_t, image_size=self.image_size)
        self.image.paste(image, box=xy)

        if anno is None:
            return

        # now draw the annotation.
        image_util.annotate(image=self.image, draw=self.draw, font=self.font,
                            text=anno, upper_left=xy, within_size=self.image_size)

    @torch.no_grad()
    def update(self, exp: Experiment, train_loss_epoch: float):
        start = datetime.datetime.now()

        row = self._current_row(exp, exp.nepochs)
        self._draw_exp_column_labels(exp)

        # update epoch label
        _row_x, row_y = self._pos_for(row=row, col=0)
        row_label = f"epoch {exp.nepochs + 1}"
        self.draw.rectangle(xy=(0, row_y, self.fixed_labels_x, row_y + self.image_size), fill='black')
        self.draw.text(xy=(0, row_y), text=row_label, font=self.font, fill='white')

        # fill out fixed column images & annotations, but only as we reach
        # that epoch.
        if row != self.last_row_fixed_drawn:
            fixed_tuples = self.generator.get_fixed_images(row)
            for fixed_img_idx, fixed_tuple in enumerate(fixed_tuples):
                x = self.fixed_labels_x + fixed_img_idx * self.image_size
                y = self.content_y + row * self.image_size
                self._draw_image_tuple(xy=(x, y), image_tuple=fixed_tuple)

            self.last_row_fixed_drawn = row

        # walk through the image/annotations we get back from the generator.
        exp_tuples = self.generator.get_exp_images(exp, row, train_loss_epoch)
        for img_idx, exp_tuple in enumerate(exp_tuples):
            col = self.exp_ncols * exp.exp_idx + img_idx
            xy = self._pos_for(row=row, col=col)
            self._draw_image_tuple(xy=xy, image_tuple=exp_tuple)

        # save image to temp path, then rename to real path
        self.image.save(self.path_temp)
        self.path_temp.rename(self.path)

        end = datetime.datetime.now()
        print(f"  updated progress image in {(end - start).total_seconds():.3f}s")

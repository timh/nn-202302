# %%
import datetime
import sys
import math
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple, Callable
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

    def get_col_labels(self) -> List[str]:
        pass

    def get_images(self, exp: Experiment, epoch: int, row: int) -> List[Tensor]:
        pass

"""
Generate a unique image for each Experiment.
It has (max_epochs // progress_every_nepochs) rows, and (ncols) columns.

If padding around images is desired, pass in a larger image_size.
"""
class ImageProgressLogger(trainer.TrainerLogger):
    progress_every_nepochs: int = 0

    _content_x = None
    _content_y = None
    _spacing_x: int
    _spacing_y: int

    _padding = 2
    _path: Path
    _img: Image.Image
    _draw: ImageDraw.ImageDraw
    _font: ImageFont.ImageFont
    _to_image: transforms.ToPILImage

    _title_xy: Tuple[int, int]

    def __init__(self,
                 dirname: str,
                 progress_every_nepochs: int,
                 image_size: Tuple[int, int],
                 generator: ImageProgressGenerator):
        super().__init__(dirname)
        self.progress_every_nepochs = progress_every_nepochs
        self.image_size = image_size
        self.generator = generator
        self._to_image = transforms.ToPILImage("RGB")
    
    def _pos_for(self, exp: Experiment, row: int, col: int) -> Tuple[int, int]:
        x = self._content_x + col * self.image_size[0]
        y = self._content_y + row * self.image_size[1]
        return x, y

    def on_exp_start(self, exp: Experiment):
        nrows = exp.max_epochs // self.progress_every_nepochs
        self.generator.on_exp_start(exp, nrows)

        self._path = Path(self._status_path(exp, "images", suffix="-progress.png"))

        # initialize the image, the image title, and the row labels.
        font_size = math.ceil(self.image_size[0] / 10)   # heuristic
        self._font = ImageFont.truetype(Roboto, font_size)

        col_labels = self.generator.get_col_labels()

        self._content_x = self.image_size[0] // 2
        self._content_y = self.image_size[0] // 8

        width = self._content_x + len(col_labels) * self.image_size[1]
        titles, title_height = image_util.experiment_labels([exp], width, self._font)
        height = title_height + self._content_y + self._padding * 2 + nrows * self.image_size[0]

        self._img = Image.new("RGB", (width, height))
        self._draw = ImageDraw.ImageDraw(self._img)
        self._title_xy = (10, int(height - title_height - self._padding * 2))
        self._draw.text(xy=self._title_xy, text=titles[0], font=self._font, fill='white')

        # draw the column labels
        for col, col_label in enumerate(col_labels):
            x, _y = self._pos_for(exp, row=0, col=col)
            self._draw.text(xy=(x, 0), text=col_label, font=self._font, fill='white')

        self._img.save(self._path)

    def on_epoch_end(self, exp: Experiment, epoch: int, train_loss_epoch: float):
        if self.progress_every_nepochs and (epoch + 1) % self.progress_every_nepochs == 0:
            start = datetime.datetime.now()

            row = epoch // self.progress_every_nepochs

            title_list, _ = image_util.experiment_labels([exp], self._img.width, self._font)
            box = (self._title_xy, (self._img.width, self._img.height))
            self._draw.rectangle(xy=box, fill='black')
            self._draw.text(xy=self._title_xy, text=title_list[0], font=self._font, fill='white')

            _row_x, row_y = self._pos_for(exp, row=row, col=0)
            row_label = f"epoch {epoch + 1}"
            self._draw.text(xy=(0, row_y), text=row_label, font=self._font, fill='white')

            img_tensors = self.generator.get_images(exp, epoch, row)
            for col, img_t in enumerate(img_tensors):
                img = self._to_image(img_t)

                xy = self._pos_for(exp, row, col)
                self._img.paste(img, box=xy)
            
            self._img.save(self._path)

            symlink_path = Path("runs", "last-progress.png")
            symlink_path.unlink(missing_ok=True)
            symlink_path.symlink_to(self._path.absolute())

            end = datetime.datetime.now()
            print(f"  updated progress image in {(end - start).total_seconds():.3f}s")



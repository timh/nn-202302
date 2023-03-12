# %%
import datetime
from collections import deque
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont

from experiment import Experiment

"""
return labels that will fit in the available_width with the given font.

return:
  List of labels
  Maximum height used
"""
def experiment_labels(experiments: List[Experiment], 
                      max_width: int,
                      font: ImageFont.ImageFont) -> Tuple[List[str], int]:
    img = Image.new("RGB", (max_width * 2, 500))
    draw = ImageDraw.ImageDraw(img)
    now = datetime.datetime.now()

    labels: List[List[str]] = list()
    heights: List[int] = list()

    for exp in experiments:
        lines = [""]

        # --
        # split the label on commas to make it fit
        # --
        remaining = exp.label
        exp_height = 0

        while len(remaining):
            if "," in remaining:
                first, rest = remaining.split(",", maxsplit=1)
                first += ", "
            else:
                first, rest = remaining, ""

            # if the line is currently empty, put the first chunk on it, no matter
            # the length.
            if not len(lines[-1]):
                if len(lines) == 1:
                    lines[-1] = first
                else:
                    # lines after the first are indented.
                    lines[-1] = "  " + first
                remaining = rest
                continue

            # consider adding the next piece onto the line
            new_line = lines[-1] + first

            # left, top, right, bottom
            left, top, right, bottom = draw.textbbox((0, 0), text=new_line, font=font)
            new_width = right - left
            if new_width > max_width:
                # if new_line was too wide, start a new line, without consuming remaining.
                lines.append("")
                continue

            # new_line fits. replace it with the longer one.
            lines[-1] = new_line
            remaining = rest

        if exp.saved_at:
            ago = int((now - exp.saved_at).total_seconds())
            ago_secs = ago % 60
            ago_mins = (ago // 60) % 60
            ago_hours = (ago // 3600)
            ago = deque([(val, desc) for val, desc in zip([ago_hours, ago_mins, ago_secs], ["h", "m", "s"])])
            while not ago[0][0]:
                ago.popleft()
            ago_str = " ".join([f"{val}{desc}" for val, desc in ago])
        else:
            ago_str = ""

        lines.extend([
            f"startlr {exp.startlr:.1E}",
            f"nepoch {exp.nepochs}",
            f"tloss {exp.lastepoch_train_loss:.3f}",
            f"vloss {exp.lastepoch_val_loss:.3f}",
        ])
        if ago_str:
            lines.append(ago_str)

        label = "\n".join(lines)
        left, top, right, bottom = draw.textbbox((0, 0), text=label, font=font)
        labels.append(label)
        heights.append(bottom - top)
    
    return labels, max(heights)

if __name__ == "__main__":
    import loadsave
    from fonts.ttf import Roboto
    from pathlib import Path

    font = ImageFont.truetype(Roboto, 10)

    checkpoints = loadsave.find_checkpoints(Path("denoise/runs"), only_paths="sd")
    exps = [cp[1] for cp in checkpoints]
    experiment_labels(exps, max_width=128, font=font)
    



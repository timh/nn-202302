from typing import List, Tuple, Dict, Callable, Union
from dataclasses import dataclass

from torch import nn

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator
from matplotlib.patches import ConnectionPatch
from IPython import display

class Plot:
    fig: Figure
    total_steps: int

    axes: plt.Axes
    yaxisscale: str
    yaxisfmt: str

    alt_axes: plt.Axes
    alt_yaxisscale: str
    alt_yaxisfmt: str
    alt_dataset: int

    _num_datasets: int                      # count of all datasets
    _num_normal_datasets: int               # number of normal datasets, that is, number that aren't "alt"
    labels: List[str]                       # labels for each dataset
    annotation_at_x: List[List[int]]        # annotation for [dataset idx][step]
    data: List[torch.Tensor]                # ongoing data for each dataset
    smoothed: List[torch.Tensor]            # for each dataset, smoothed value at the given point
    _steps_so_far: List[int]                # steps so far for each dataset

    def __init__(self, total_steps: int, labels: List[str], fig: Figure, nrows=1, ncols=1, idx=1, alt_dataset=-1, 
                 yaxisscale="log", alt_yaxisscale="log",
                 yaxisfmt=".5f", alt_yaxisfmt=".5f"):
        self.fig = fig
        self.total_steps = total_steps
        self._num_datasets = len(labels)
        self._num_normal_datasets = self._num_datasets - (0 if (alt_dataset == -1) else 1)

        all_datasets = range(self._num_datasets)
        all_normal_datasets = range(self._num_normal_datasets)
        self.labels = labels.copy()
        self.annotation_at_x = [list() for _ in all_datasets]
        self._steps_so_far = [0 for _ in all_datasets]

        self.data = [torch.zeros((self.total_steps, )) for _ in all_datasets]
        self.smoothed = [torch.zeros((self.total_steps, )) for _ in all_datasets]

        self.axes = self.fig.add_subplot(nrows, ncols, idx)
        self.yaxisscale = yaxisscale
        self.yaxisfmt = yaxisfmt

        self.alt_dataset = alt_dataset
        self.alt_yaxisscale = alt_yaxisscale
        self.alt_yaxisfmt = alt_yaxisfmt
        if self.alt_dataset != -1:
            self.alt_axes = self.axes.twinx()
    
    def add_data(self, idx: int, data: torch.Tensor, annotate = False):
        start = self._steps_so_far[idx]
        end = start + len(data)
        # print(f"  \033[1;33madd_data: {idx=} {start=} {end=}\033[0m")
        if end > len(self.data[idx]):
            print(f"add_data: too much data: {idx=} {start=} {end=} {self.data[idx].shape=}")
        self.data[idx][start:end] = data

        if annotate:
            self.annotation_at_x[idx].append(end - 1)

        self._steps_so_far[idx] += len(data)

    def _data_so_far(self, idx: int) -> torch.Tensor:
        return self.data[idx][:self._steps_so_far[idx]]

    def render(self, ymax_quantile: float = 0.0, smooth_steps: int = 0):
        for axes, yaxisscale in zip([self.axes, self.alt_axes], [self.yaxisscale, self.alt_yaxisscale]):
            if axes is None:
                continue
            axes.clear()
            axes.set_yscale(yaxisscale)

        # set limit of each normal axes based on chopping off the top (ymax_quantile) quantile
        if ymax_quantile:
            minval = 0.0
            maxval = 0.0
            for idx in range(self._num_normal_datasets):
                data = self._data_so_far(idx)
                if len(data) == 0:
                    continue

                ymax_quantile = torch.tensor(ymax_quantile)
                mina = torch.min(data)
                maxa = torch.quantile(data, q=ymax_quantile)
                if idx == 0:
                    minval = mina
                    maxval = maxa
                else:
                    minval = torch.minimum(minval, mina)
                    maxval = torch.maximum(maxval, maxa)

            self.axes.set_ylim(bottom=minval, top=maxval)

        for idx in range(self._num_normal_datasets):
            self._render_axes(idx, self.axes, smooth_steps)
        
        if self.alt_axes is not None:
            self._render_axes(self.alt_dataset, self.alt_axes, 0)
        
        all_lines = []
        all_labels = []
        for axes in [self.axes] + ([self.alt_axes] if self.alt_axes is not None else []):
            lines, labels = axes.get_legend_handles_labels()
            all_lines.extend(lines)
            all_labels.extend(labels)
        self.axes.legend(all_lines, all_labels)
    
    def _render_axes(self, idx: int, axes: plt.Axes, smooth_steps: int):
        label = self.labels[idx]
        data = self._data_so_far(idx)

        if len(data) == 0:
            return

        # apply moving average.
        if smooth_steps >= 2:
            orig_len = len(data)

            # pool1d needs a 2d tensor
            smoothed = F.avg_pool1d(data.view((1, -1)), kernel_size=smooth_steps, stride=1, padding=smooth_steps//2, ceil_mode=True, count_include_pad=False)

            # result from avg_pool1d is 2d. turn it back into 1d. it is (smooth_steps) larger, which we chop off
            smoothed = smoothed.view((smoothed.shape[-1], ))[:orig_len]

            # keep track of this smoothed result
            self.smoothed[idx][0:len(smoothed)] = smoothed
        else:
            smoothed = None
        
        # plot the data.
        kwargs = {}
        if idx == self.alt_dataset:
            pltres = axes.plot(data, label=label, linestyle='dashed', color='black')
        else:
            if smoothed is not None:
                pltres = axes.plot(smoothed, label=label + f" smooth {smooth_steps}")
                # axes.plot(data, label=label, linestyle='dotted', color=pltres[0].get_color())
            else:
                pltres = axes.plot(data, label=label)
        
        pltcolor = pltres[0].get_color()

        for annox in self.annotation_at_x[idx]:
            # put the annotations above the maximum value for any dataset except
            # the last (shared) one. use both raw and smoothed values to figure
            # out the max.
            if idx == self.alt_dataset:
                max_at_annox = data[annox]
                xoff, yoff = 2, 10
                yaxisfmt = self.alt_yaxisfmt
            else:
                if smoothed is not None:
                    max_at_annox = max([s[annox] for s in self.smoothed])
                else:
                    max_at_annox = max([d[annox] for d in self.data[:-1]])
                xoff, yoff = 2, 10 + 15 * idx
                yaxisfmt = self.yaxisfmt

            xy = (annox, max_at_annox)
            text = format(data[annox], yaxisfmt)

            axes.annotate(text=text, xy=xy, label=label, color=pltcolor,
                          xytext=(xoff, yoff), textcoords='offset pixels')

        return self.fig

def imshow(net: nn.Module, fn: Callable[[torch.nn.Linear], torch.Tensor],
           fig: Figure, fn_fig: Callable[[int, int], Figure],
           nrows: int, row: int,
           vmin: float=None, vmax: float=None,
           title="") -> Tuple[Figure, Axes]:
    tensors = [fn(m) for m in net.modules() if isinstance(m, torch.nn.Linear)]
    count_all_rows = max([t.shape[0] for t in tensors])
    count_all_cols = sum([t.shape[1] for t in tensors])

    all_tensors = torch.zeros((count_all_rows, count_all_cols))
    running_col = 0
    for t in tensors:
        trows, tcols = t.shape
        all_tensors[:trows, running_col:running_col+tcols] = t
        running_col += tcols

    if fig is None:
        fig = fn_fig(count_all_rows, count_all_cols)

    kvargs = {}
    if vmin is not None:
        kvargs['vmin'] = vmin
    if vmax is not None:
        kvargs['vmax'] = vmax

    # make this subplot the fig width of parent fig, and the height the appropriate 
    # height depending on the dimensions of the data.
    axes: Axes = None
    for ax in fig.axes:
        ax_nrows, ax_ncols, ax_start, ax_stop = ax.get_subplotspec().get_geometry()
        # NOTE that get_geometry() returns index starting at 0, whereas add_subplot
        # starts at 1.
        if ax_nrows == nrows and ax_ncols == 1 and ax_start == row - 1:
            axes = ax
            axes.clear()
            break

    if axes is None:
        axes = fig.add_subplot(nrows, 1, row)

    if title:
        axes.set_title(title)
    
    axes.imshow(all_tensors.detach().cpu(), cmap=matplotlib.cm.gray, **kvargs)

    return fig, axes

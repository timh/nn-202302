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
    axes: List[plt.Axes]
    total_epochs: int

    annotation_at_x: List[List[int]]
    data: List[torch.Tensor]                # ongoing data for each dataset
    smoothed: List[torch.Tensor]            # for each dataset, smoothed value at the given point
    _epochs_so_far: List[int]

    def __init__(self, total_epochs: int, labels: List[str], fig: Figure, nrows=1, ncols=1, idx=1):
        self.fig = fig
        self.annotation_at_x = [list() for _ in labels]
        self.total_epochs = total_epochs
        self._epochs_so_far = [0 for _ in labels]
        self.labels = labels.copy()
        self.data = [torch.zeros((self.total_epochs, )) for _ in labels]
        self.smoothed = [torch.zeros((self.total_epochs, )) for _ in labels[:-1]]
        self.axes = list()
        self.axes.append(self.fig.add_subplot(nrows, ncols, idx))
        self.axes.append(self.axes[0].twinx())
    
    def add_data(self, idx: int, data: torch.Tensor):
        start = self._epochs_so_far[idx]
        end = start + len(data)
        if end > len(self.data[idx]):
            print(f"add_data: too much data: {start=} {end=} {self.data[idx].shape=}")
        self.data[idx][start:end] = data

        self._epochs_so_far[idx] += len(data)

    def _data_so_far(self, idx: int) -> torch.Tensor:
        return self.data[idx][:self._epochs_so_far[idx]]

    def render(self, ylim: float = 0.0, smooth_steps: int = 0, annotate: bool = False):
        for axes in self.axes:
            axes.clear()
            axes.set_yscale("log")

            # axes.set_label(label)
            axes.yaxis.set_major_locator(LogLocator(subs='all'))
            axes.yaxis.set_minor_locator(LogLocator(subs='all'))

        if ylim:
            minval = 0.0
            maxval = 0.0
            for idx in range(len(self.labels[:-1])):
                data = self._data_so_far(idx)
                if len(data) == 0:
                    continue
                quantile = torch.tensor(ylim)
                mina = torch.min(data)
                maxa = torch.quantile(data, q=quantile)
                if idx == 0:
                    minval = mina
                    maxval = maxa
                else:
                    minval = torch.minimum(minval, mina)
                    maxval = torch.maximum(maxval, maxa)

            self.axes[0].set_ylim(bottom=minval, top=maxval)

        for idx in range(len(self.labels)):
            if idx == len(self.labels) - 1:
                self._render_axes(idx, self.axes[1], 0, False)
            else:
                self._render_axes(idx, self.axes[0], smooth_steps, annotate)
        
        all_lines = []
        all_labels = []
        for axes in self.axes:
            lines, labels = axes.get_legend_handles_labels()
            all_lines.extend(lines)
            all_labels.extend(labels)
        self.axes[0].legend(all_lines, all_labels)
    
    def _render_axes(self, idx: int, axes: plt.Axes, smooth_steps: int, annotate: bool):
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
        
        if annotate:
            self.annotation_at_x[idx].append(len(data) - 1)

        # plot the data.
        kwargs = {}
        if idx == len(self.labels) - 1:
            pltres = axes.plot(data, label=label, linestyle='dashed', color='black')
        else:
            if smoothed is not None:
                pltres = axes.plot(smoothed, label=label + f" smooth {smooth_steps}")
                # axes.plot(data, label=label, linestyle='dotted', color=pltres[0].get_color())
            else:
                pltres = axes.plot(data, label=label)
        
        pltcolor = pltres[0].get_color()

        for annox in self.annotation_at_x[idx]:
            # put the annotations above the maximum value for any dataset. use both 
            # raw and smoothed values to figure out the max.
            if smoothed is not None:
                max_at_annox = max([s[annox] for s in self.smoothed])
            else:
                max_at_annox = max([d[annox] for d in self.data[:-1]])
            xy = (annox, max_at_annox)
            text = f"{data[annox]:.5f}"
            xoff, yoff = 2, 10 + 15 * idx

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
        # print(f"trows {trows}, tcols {tcols}")
        # print(f"[:{trows}, {running_col}:{running_col+tcols}] = {t.shape}")
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

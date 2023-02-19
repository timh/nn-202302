from typing import List, Tuple, Dict
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator
from matplotlib.patches import ConnectionPatch
from IPython import display

@dataclass
class Annotation:
    text: str
    xy: Tuple[float, float]
    axes: str
    pos: str

class Plot:
    fig: Figure
    annotations: List[Annotation]
    labels: List[str]
    colors: List[str]
    data: List[torch.Tensor]
    axes: List[plt.Axes]
    total_epochs: int
    _epochs_so_far: int

    def __init__(self, total_epochs: int, labels: List[str], colors: List[str], fig: Figure, nrows=1, ncols=1, idx=1):
        self.fig = fig
        self.annotations = list()
        self.total_epochs = total_epochs
        self._epochs_so_far = 0
        self.labels = labels.copy()
        self.colors = colors.copy()
        self.data = list()
        self.axes = list()
        if len(labels) not in [1, 2]:
            raise ValueError("only 1 or two data labels allowed")
        
        self.data.append(torch.zeros((self.total_epochs, )))
        self.axes.append(self.fig.add_subplot(nrows, ncols, idx))
        if len(labels) > 1:
            self.data.append(torch.zeros_like(self.data[0]))
            self.axes.append(self.axes[0].twinx())
    
    def add_data(self, data_list: List[torch.Tensor]):
        for i, data in enumerate(data_list):
            start = self._epochs_so_far
            end = start + len(data)
            self.data[i][start:end] = data

        self._epochs_so_far += len(data_list[0])

    def add_annotation(self, idx: int, xy: Tuple[float, float], annotext: str, pos: str):
        # add an annotation on the given axis, given xy
        a = Annotation(annotext, xy, axes=self.labels[idx], pos=pos)
        self.annotations.append(a)
    
    def _data_so_far(self, idx: int) -> torch.Tensor:
        return self.data[idx][:self._epochs_so_far]

    def render(self, ylim: float = 0.0, smooth_steps: int = 0, annotate: bool = False) -> plt.Figure:
        for idx, (axes, label, color) in enumerate(zip(self.axes, self.labels, self.colors)):
            axes.clear()
            axes.set_yscale("log")
            axes.set_label(label)
            axes.set_ylabel(label, color=color)
            axes.yaxis.set_major_locator(LogLocator(subs='all'))
            axes.yaxis.set_minor_locator(LogLocator(subs='all'))

            if idx == 0 and ylim != 0.0:
                data = self._data_so_far(0)
                quantile = torch.tensor(ylim)
                minval = torch.min(data)
                maxval = torch.quantile(data, q=quantile)
                axes.set_ylim(bottom=minval, top=maxval)

            if idx == 0:        
                self._render_axes(idx, smooth_steps, annotate)
            else:
                self._render_axes(idx, 0, False)

        # plot the legend.
        if len(self.labels) == 2:
            lines0, labels0 = self.axes[0].get_legend_handles_labels()
            lines1, labels1 = self.axes[1].get_legend_handles_labels()
            axes.legend(lines0 + lines1, labels0 + labels1)
    
    def _render_axes(self, idx: int, smooth_steps: int, annotate: bool):
        color = self.colors[idx]
        label = self.labels[idx]
        axes = self.axes[idx]
        data = self._data_so_far(idx)

        # apply moving average.
        if smooth_steps >= 2:
            orig_len = len(data)

            # pool1d needs a 2d tensor
            data = F.avg_pool1d(data.view((1, -1)), kernel_size=smooth_steps, stride=1, padding=smooth_steps//2, ceil_mode=True, count_include_pad=False)

            # result from avg_pool1d is 2d. turn it back into 1d. it is (smooth_steps) larger, which we chop off
            data = data.view((data.shape[-1], ))[:orig_len]
        
        if annotate:
            val = data[-1]
            xy = (len(data) - 1, val)
            self.annotations.append(Annotation(f"{label} {val:.5f}", xy, label, "above"))

        # plot the data.
        axes.plot(data, color=self.colors[idx], label=label)

        for anno in self.annotations:
            kvargs = {}
            if label != anno.axes:
                continue

            if anno.pos == "below":
                xoff = 2
                yoff = (anno.text.count("\n") + 1) * -55
                kvargs = dict(xytext=(xoff, yoff), textcoords='offset pixels')

            elif anno.pos == "title":
                xoff = 150
                yoff = 10
                kvargs = dict(xytext=(xoff, yoff), textcoords='offset pixels', xycoords='axes pixels')

            else:
                xoff, yoff = -60, 30
                kvargs = dict(xytext=(xoff, yoff), textcoords='offset pixels', arrowprops=dict(arrowstyle='->'))

            axes.annotate(text=anno.text, xy=anno.xy, color="black", **kvargs)

        return self.fig

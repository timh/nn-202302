from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List
import numpy as np
from fonts.ttf import Roboto

from layer import Layer, DenseLayer, ActivationReLU
from network import Network
import io, sys

FORMAT = "9.4f"

def draw_step(network: Network, out: io.BufferedWriter):
    def render_row(vals: np.ndarray, classRow: str) -> None:
        val_strs = format_floats(vals)
        print(f"<span class=\"{classRow}\">", file=out)
        for val in val_strs:
            print(f"  <span class=\"item\">{val}</span>", file=out)
        print(f"</span> <!-- {classRow} -->", file=out)

    def render_2d(rows: np.ndarray, classOverall: str, classRow: str) -> None:
        # rendering array of array of strings, in reverse order.
        print(f"<span class=\"{classOverall}\">", file=out)
        for row in reversed(rows):
            render_row(row, classRow)
        print(f"</span> <!-- {classOverall} -->", file=out)

    def format_floats(vals: np.ndarray, heading: str = "") -> List[str]:
        res: List[str] = list()
        if heading:
            res.append(heading)

        res.extend([format(v, FORMAT) for v in vals])
        return res
    
    print("<div class=\"network\">", file=out)

    for lidx, layer in enumerate(network.layers):
        num_neurons = layer.weights.shape[1]

        print("<div class=\"layer\">", file=out)

        if isinstance(layer, DenseLayer):
            for nidx in range(num_neurons):
                print("<div class=\"neuron\">", file=out)

                render_row(layer.inputs.T[nidx], "inputs")
                render_row(layer.dinputs.T[nidx], "dinputs")

                # weights & biases
                render_row(layer.weights.T[nidx], "weights")
                render_row(layer.dweights.T[nidx], "dweights")

                render_row(layer.biases.T[nidx], "biases")
                render_row(layer.dbiases.T[nidx], "dbiases")

                render_row(layer.outputs.T[nidx], "outputs")

                print("</div> <!-- neuron -->", file=out)
        else:
            inputs_T = layer.inputs.T
            dinputs_T = layer.dinputs.T
            for iidx in range(inputs_T.shape[0]):
                render_row(inputs_T[iidx], "inputs")
                render_row(dinputs_T[iidx], "dinputs")

            outputs_T = layer.outputs.T
            for oidx in range(outputs_T.shape[0]):
                render_row(outputs_T[oidx], "inputs")

        render_row(layer.outputs.T[nidx], "outputs")

        print("</div> <!-- layer -->", file=out)

    print("</div> <!-- network -->", file=out)

from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List
import numpy as np
from fonts.ttf import Roboto

from layer import Layer
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

    net_input = network.layers[0].last_input
    render_2d(net_input, "inputs", "input-row")

    for lidx, layer in enumerate(network.layers):
        num_neurons = layer.weights.shape[1]

        print("<div class=\"layer\">", file=out)

        for nidx in range(num_neurons):
            print("<div class=\"neuron\">", file=out)

            # weights & biases
            weights = layer.save_weights.T[nidx]
            mult_weights = layer.save_weights.T[nidx] * net_input[0]
            render_row(weights, "weights")
            render_row(mult_weights, "res-weights")
            render_row(layer.deriv_mult_weights.T[nidx], "deriv-weights")

            bias = layer.save_biases.T[nidx]
            render_row(bias, "bias")
            render_row(layer.last_sum.T[nidx], "res-sum")
            render_row(layer.deriv_sum_bias.T[nidx], "deriv-sum")

            render_row(layer.last_result.T[nidx], "res-relu")
            render_row(layer.deriv_relu.T[nidx], "deriv-relu")

            print("</div> <!-- neuron -->", file=out)

        print("</div> <!-- layer -->", file=out)

    print("</div> <!-- network -->", file=out)
    print("</body>", file=out)
    print("</html>", file=out)

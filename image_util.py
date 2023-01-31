from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List
import numpy as np
from fonts.ttf import Roboto

from layer import Layer
from network import Network

ROBOTO = ImageFont.truetype(Roboto, 15)
BORDER = 2
FONT_VERT_SPACING = 2
FORMAT = "9.4f"

class ImageStepDesc:
    image: Image.Image
    draw: ImageDraw.ImageDraw
    font: ImageFont.ImageFont
    network: Network

    def __init__(self, network: Network):
        self.network = network
        self.image = Image.new(mode="RGB", size=(1024, 512), color="white")
        self.draw = ImageDraw.Draw(self.image)
        self.font = ROBOTO

    def draw_step(self):
        def draw_multidim(vals: np.ndarray, xy: Tuple[int, int], color: str, border = False) -> Tuple[int, int]:
            maxy = xy[1]
            x = xy[0]
            y = xy[1]
            for column in reversed(vals):
                one_col_array = []
                for value in column:
                    one_col_array.append(format(value, FORMAT))
                one_input_str = "\n".join(one_col_array)
                xy_input = draw_multiline(one_input_str, (x, y), color, self.draw, self.font, border)
                x = xy_input[0]
                maxy = max(maxy, xy_input[1])

            return x, maxy
        
        def format_floats(vals: np.ndarray, heading: str = "") -> List[str]:
            res: List[str] = list()
            if heading:
                res.append(heading)

            res.extend([format(v, FORMAT) for v in vals])
            return res

        net_input = self.network.layers[0].last_input
        x, _ = draw_multidim(net_input, (0, 0), "blue", True)
        x += 20

        for lidx, layer in enumerate(self.network.layers):
            layer: Layer = layer
            num_neurons = layer.weights.shape[1]
            y = 0
            for nidx in range(num_neurons):
                # weights & biases
                weights_biases = format_floats(layer.weights.T[nidx])
                weights_biases.append("    " + format(layer.biases.T[nidx][0], FORMAT))
                right, bot = draw_multiline(weights_biases, (x, y), "black", self.draw, self.font, True)

                # summed weights - no bias
                sum_nb_res = layer.last_sum_no_bias[nidx]
                sum_nb_str = "\n".join(format_floats(sum_nb_res, "sum"))
                right, bot = draw_multiline(sum_nb_str, (right, y), "green", self.draw, self.font, True)

                # summed weights - with bias
                sum_res = layer.last_sum[nidx]
                sum_str = "\n".join(format_floats(sum_res, "sum+b"))
                right, bot = draw_multiline(sum_str, (right, y), "green", self.draw, self.font, True)

                # output of RELU
                relu_res = layer.last_result[nidx]
                relu_str = "\n".join(format_floats(relu_res, "relu"))
                right, bot = draw_multiline(relu_str, (right, y), "green", self.draw, self.font, True)

                y = bot + 20
        
            x = right + 50

# returns right, bottom of text drawn.
def draw_multiline(text: str, xy: Tuple[int, int], color: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, border = False) -> Tuple[int, int]:
    x = xy[0]
    y = xy[1]
    if border:
        x += BORDER
        y += BORDER

    if isinstance(text, str):
        input_strs = text.split("\n")
    else:
        input_strs = text

    maxx = 0
    maxy = 0
    for line_no, input_str in enumerate(input_strs):
        bbox = font.getbbox(input_str)
        text_width = bbox[2] - bbox[0]
        # text_height = bbox[3] - bbox[1]
        text_height = font.size

        draw.text(xy=(x, y), text=input_str, font=font, fill=color)
        y += text_height + FONT_VERT_SPACING
        maxx = max(maxx, x + text_width)
        maxy = max(maxy, y)
    
    if border:
        maxx += BORDER
        maxy += BORDER
        draw.rectangle((xy[0], xy[1], maxx, maxy), outline=color)

    return (maxx, maxy)
    


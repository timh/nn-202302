from PIL import Image, ImageDraw, ImageFont
from typing import Tuple
import numpy as np
from fonts.ttf import Roboto

from layer import Layer

ROBOTO = ImageFont.truetype(Roboto, 10)

class Network: pass
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

    def draw_forward(self):
        input = self.network.layers[0].last_input

        x = 0
        for input_one in reversed(input):
            one_input_array = []
            for value in input_one:
                one_input_array.append(f"{value:10.4f}")
            one_input_str = "\n".join(one_input_array)
            xy_input = draw_multiline(one_input_str, (x, 0), "blue", self.draw, self.font)
            x = xy_input[0]
        
        x += 20

        for lidx, layer in enumerate(self.network.layers):
            num_neurons = layer.weights.shape[1]
            maxx = x
            neuron_tops = list()
            neuron_bots = list()
            y = 0
            border = 2
            for nidx in range(num_neurons):
                weights = [format(v, ".4f") for v in layer.weights.T[nidx]]
                bias = format(layer.biases.T[nidx][0], ".4f")

                weight_bbox = draw_multiline(weights, (x + border, y + border), "black", self.draw, self.font)
                biasx = (x + weight_bbox[0]) / 2
                biasy = weight_bbox[1]
                bias_bbox = draw_multiline(bias, (biasx, biasy), "black", self.draw, self.font)

                top = y
                bot = bias_bbox[1] + border
                self.draw.rectangle((x, y, bias_bbox[0], bot), outline="black")

                neuron_tops.append(top)
                neuron_bots.append(bot)
                y = bot + 100
                maxx = max(maxx, bias_bbox[0])
            
            x = maxx + 10
            for iidx, one_res in enumerate(layer.last_result.T):
                y = neuron_tops[iidx]
                one_res_str = "\n".join([format(v, ".4f") for v in one_res])
                res_bbox = draw_multiline(one_res_str, (x, y), "green", self.draw, self.font)
                maxx = max(maxx, x + res_bbox[0])
        
            x = maxx + 20

# returns right, bottom of text drawn.
def draw_multiline(text: str, xy: Tuple[int, int], color: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> Tuple[int, int]:
    font_height = font.size + 2

    x = 0
    y = 0

    if isinstance(text, str):
        input_strs = text.split("\n")
    else:
        input_strs = text

    maxx = 0
    maxy = 0
    for line_no, input_str in enumerate(input_strs):
        textbox = font.getbbox(input_str)
        text_width = textbox[2] - textbox[0]
        x = xy[0]
        y = xy[1] + line_no * font_height
        maxx = max(maxx, x + text_width)
        maxy = max(maxy, y + font_height)

        draw.text(xy=(x, y), text=input_str, font=font, fill=color)
    
    return (maxx, maxy)
    


# %%
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
vis_model = model.vision_model.to("cuda")
text_model = model.text_model.to("cuda")
print(f"{model.text_embed_dim=}")
print(f"{model.vision_embed_dim=}")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# model.

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
inputs = inputs.to("cuda")

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

inputs = processor(images=image, return_tensors="pt", padding=True).to("cuda")
im_feat = model.get_image_features(output_hidden_states=True, return_dict=True, **inputs)
print("im_feat:", im_feat)
print("im_feat:", im_feat.shape)

tokens = tokenizer("a photo of a cat", padding="max_length", return_tensors="pt").input_ids
tokens = tokens.to("cuda")
print(f"{tokens=}")
embeds = text_model(tokens)
print(f"{embeds[0].shape=}")


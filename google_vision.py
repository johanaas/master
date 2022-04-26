import io
import os

from google.cloud import vision

client = vision.ImageAnnotatorClient()

file_name = os.path.abspath(r"C:\Users\kamidtli\dev\ILSVRC2012_img_val\ILSVRC2012_val_00000009.JPEG")

with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = vision.Image(content=content)

response = client.label_detection(image=image)
labels = response.label_annotations

print("Labels:")
for label in labels:
    print(label.description)

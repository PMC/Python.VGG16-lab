from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image
import numpy as np

model = VGG16(weights="imagenet")
img_path = "img/avatar.png"
# img_path = "img/snowman.jpg"

myimage = image.load_img(img_path, target_size=(224, 224))
# myimage.show()

image_array = image.img_to_array(myimage)

x = np.expand_dims(image_array, axis=0)
x = preprocess_input(x)

preds = model.predict(x)

# Decode the predictions
top_n = 10
decoded = decode_predictions(preds, top=top_n)[0]
total_prob = sum(prob for (_, _, prob) in decoded)
for i, (_, label, prob) in enumerate(decoded):
    norm_prob = prob / total_prob
    print(f"{i + 1}: {label} ({norm_prob * 100:.2f}%)")

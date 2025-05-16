from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image
import numpy as np

model = VGG16(weights="imagenet")
img_path = "img/avatar.png"

myimage = image.load_img(img_path, target_size=(224, 224))
# myimage.show()

image_array = image.img_to_array(myimage)

x = np.expand_dims(image_array, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
decoded = decode_predictions(preds, top=3)[0]
for i, (imagenet_id, label, prob) in enumerate(decoded):
    print(f"{i + 1}: {label} ({prob * 100:.2f}%)")

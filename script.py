import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('weights.h5')

img_path = '0003.JPEG'

img = image.load_img(img_path, target_size=(125, 125))

x = image.img_to_array(img)

x = x / 255.0

x = np.expand_dims(x, axis=0)

prediction = model.predict(x)


if prediction < 0.5:
    print("The car is damaged")
else:
    print("The car is  not damaged")

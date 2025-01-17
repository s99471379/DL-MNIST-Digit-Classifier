import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = train_x /255.0
test_x = test_x / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10) 
])
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs=10)

test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)
print('Test accuracy:', test_acc)

predictions = model.predict(test_x)
model.save("mnist.h5")
new_model = keras.models.load_model("mnist.h5")


image_path = "unknown_digits/5.png"
new_image = image.load_img(image_path, target_size=(28, 28), color_mode='grayscale')
img = np.array(new_image)
img = np.expand_dims(img,0)
img = img / 255.0
predict = np.argmax(new_model.predict(img), axis=-1)


plt.imshow(new_image, cmap=plt.cm.binary)
plt.title(predict)
plt.show()

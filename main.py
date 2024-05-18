# Resource: https://www.tensorflow.org/tutorials/images/classification

# Importing libraries
import matplotlib.pyplot as plt  # Plotting graphs and visualizing data
import numpy as np  # Numerical operations and array manipulation
import pathlib  # Handling file paths
import tensorflow as tf  # Building and training CNNs
from PIL import Image  # Opening and manipulating images

# Setting up paths and parameters
data_dir = pathlib.Path('birds')
print(list(data_dir.glob('*')))  # From pathlib library, lists all subdirectories in the data directory
batch_size = 80
img_height = 180
img_width = 180

# Total of 800 images

# Training the Dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,  # Specify that 20% of the data from the dataset is reserved for validation
    subset="training",  # Indicates that this dataset is for training purposes
    seed=123,  # Ensures consistent results in random operations
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Validating the Dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",  # Indicates that this dataset is for validation purposes
    seed=123,  # Ensures consistent results in random operations
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Loading a random image of a Cassowary
cassowary = list(data_dir.glob('cassowary/*'))
print("NUMBER OF CASSOWARY IMAGES:", len(cassowary))
if cassowary:
    image = Image.open(str(cassowary[11]))
    image.show()

# Loading a random image of an Emu
emu = list(data_dir.glob('emu/*'))
print("NUMBER OF EMU IMAGES:", len(emu))
if emu:
    image = Image.open(str(emu[19]))
    image.show()

# Loading a random image of an Ostrich
ostrich = list(data_dir.glob('ostrich/*'))
print("NUMBER OF OSTRICH IMAGES:", len(ostrich))
if ostrich:
    image = Image.open(str(ostrich[187]))
    image.show()

# Loading a random image of a Rhea
rhea = list(data_dir.glob('rhea/*'))
print("NUMBER OF RHEA IMAGES:", len(rhea))
if rhea:
    image = Image.open(str(rhea[6]))
    image.show()

class_names = train_ds.class_names
print("CLASS NAMES:", class_names)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal",
                               input_shape=(img_height, img_height, 3)),  # Randomly flip images horizontally
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Build the model with input shape defined at the start of the model
model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names))
])

model.summary()

# Adaptive Movement Estimation
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Training the Model
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Visualizing the training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), acc, label='Training Accuracy')
plt.plot(range(epochs), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(epochs), loss, label='Training Loss')
plt.plot(range(epochs), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Predict an Image
img_path = "test/test_3.jpg"  # test_1(Cassowary), test_2(Emu), test_3(Ostrich), test_4(Rhea)
img = tf.keras.utils.load_img(
    img_path, target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(f"This image most likely belongs to {class_names[np.argmax(score)]} "
      f"with a {100 * np.max(score):.2f} percent confidence.")

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
TF_MODEL_FILE_PATH = 'model.tflite'  # The default path to the saved TensorFlow Lite model

# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Converted model is saved.")

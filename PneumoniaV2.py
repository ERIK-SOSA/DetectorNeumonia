# import os
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
# from tensorflow import keras
# from sklearn.model_selection import train_test_split
# from tensorflow.keras import layers

# # ruta de la carpeta de imágenes de neumonía
# folder_pneumonia = "chest_xray/train/PNEUMONIA"
# # lista de todas las imágenes de neumonía en la carpeta
# images_pneumonia = os.listdir(folder_pneumonia)

# print(images_pneumonia)

# # ruta de la carpeta de imágenes normales
# folder_normal = "chest_xray/train/NORMAL"
# # lista de todas las imágenes normales en la carpeta
# images_normal = os.listdir(folder_normal)

# # crea una lista vacía para almacenar las imágenes y sus etiquetas
# data = []
# labels = []

# # carga las imágenes y sus etiquetas en la lista de datos
# for image_name in images_pneumonia:
#     # carga la imagen de neumonía y la etiqueta "PNEUMONIA"
#     img = Image.open(os.path.join(folder_pneumonia, image_name))
#     img_array = np.array(img)
#     data.append(img_array)
#     labels.append(1) # 1 = PNEUMONIA

# for image_name in images_normal:
#     # carga la imagen normal y la etiqueta "NORMAL"
#     img = Image.open(os.path.join(folder_normal, image_name))
#     img_array = np.array(img)
#     data.append(img_array)
#     labels.append(0) # 0 = NORMAL

# # convierte las listas de datos y etiquetas en matrices numpy
# data = np.array(data, dtype=object)
# labels = np.array(labels)

# # normaliza los datos (escala los valores de píxeles entre 0 y 1)
# data = data / 255.0

# # divide los datos en conjuntos de entrenamiento y prueba
# train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# # define el modelo de red neuronal
# model = keras.Sequential([
#     layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(150, 150, 3)),
#     layers.MaxPooling2D(),
#     layers.Conv2D(32, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(64, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])

# # compila el modelo
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # entrena el modelo
# model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

# # evalúa el modelo en los datos de prueba
# test_loss, test_acc = model.evaluate(test_data, test_labels)
# print('Test accuracy:', test_acc)


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Se definen los parámetros del modelo
input_shape = (150, 150, 3)
batch_size = 32
num_classes = 2
epochs = 20

# Se crea un objeto ImageDataGenerator para hacer el preprocesamiento de las imágenes
train_datagen = ImageDataGenerator(rescale=1./255)

# Se carga el conjunto de entrenamiento
train_generator = train_datagen.flow_from_directory(
        'chest_xray/train',
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical')

# Se crea el modelo CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Se compila el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Se entrena el modelo
model.fit(train_generator, epochs=epochs)


# Guardar el modelo
model.save('pneumonia_model.h5')
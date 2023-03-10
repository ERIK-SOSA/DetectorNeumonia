# import os
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt

# # ruta de la carpeta de imágenes
# folder = "chest_xray/train/NORMAL"

# # lista de todas las imágenes en la carpeta
# images = os.listdir(folder)

# # muestra las primeras 4 imágenes
# for i in range(4):
#     # carga la imagen
#     img = Image.open(os.path.join(folder, images[i]))
#     # convierte la imagen en una matriz de píxeles
#     img_array = np.array(img)
#     # muestra la imagen y su etiqueta
#     plt.imshow(img_array, cmap="gray")
#     plt.title("NORMAL")
#     plt.show()




import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ruta de la carpeta de imágenes de neumonía
folder_pneumonia = "chest_xray/train/PNEUMONIA"
# lista de todas las imágenes de neumonía en la carpeta
images_pneumonia = os.listdir(folder_pneumonia)

# ruta de la carpeta de imágenes normales
folder_normal = "chest_xray/train/NORMAL"
# lista de todas las imágenes normales en la carpeta
images_normal = os.listdir(folder_normal)

# muestra las primeras 4 imágenes de neumonía y las primeras 4 imágenes normales
for i in range(4):
    # carga la imagen de neumonía
    img_pneumonia = Image.open(os.path.join(folder_pneumonia, images_pneumonia[i]))
    # convierte la imagen en una matriz de píxeles
    img_array_pneumonia = np.array(img_pneumonia)
    # muestra la imagen y su etiqueta
    plt.imshow(img_array_pneumonia, cmap="gray")
    plt.title("PNEUMONIA")
    plt.show()

    # carga la imagen normal
    img_normal = Image.open(os.path.join(folder_normal, images_normal[i]))
    # convierte la imagen en una matriz de píxeles
    img_array_normal = np.array(img_normal)
    # muestra la imagen y su etiqueta
    plt.imshow(img_array_normal, cmap="gray")
    plt.title("NORMAL")
    plt.show()
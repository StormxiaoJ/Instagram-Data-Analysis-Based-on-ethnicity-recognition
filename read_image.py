import os
import numpy as np
from PIL import Image
import h5py
import cv2
import face_recognition

# Folder path
PATH = ["Caucasian", "African", "Indian", "Mongoloid"]
# label is 0-3. 0= caucasian, 1=african, 2=indian, 3=mongoloid
images = []
ethnic = []

for i in range(len(PATH)):
    for dirpath, dirs, files in os.walk("dataset\\" + PATH[i]):
        for file in files:
            if file.endswith(".jpg"):
                name = os.path.join(dirpath, file)
                image = face_recognition.load_image_file(name)
                face_locations = face_recognition.face_locations(image)
                if len(face_locations) != 1:
                    pass
                top = face_locations[0][0]
                right = face_locations[0][1]
                bottom = face_locations[0][2]
                left = face_locations[0][3]
                if bottom - top < 80 or right - left < 80:
                    pass
                images.append(cv2.resize(image[top:bottom + 1, left:right + 1, :], (160, 160)))
                ethnic.append(i)
# Concatenate
images = np.float64(np.stack(images))
print(images.shape)
ethnic = np.stack(ethnic)

# Save to disk
f = h5py.File("images.h5", "w")
# Create images dataset
X_dset = f.create_dataset('data', images.shape, dtype='f')
X_dset[:] = images

# Create annotation
y_dset = f.create_dataset('ethnic', ethnic.shape, dtype='i')
y_dset[:] = ethnic

f.close()

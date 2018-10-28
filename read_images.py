import os
import numpy as np
from PIL import Image
import h5py
import cv2

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
                im = cv2.imread(name)
                images.append(np.array(im))
                ethnic.append(i)
# Concatenate
images = np.float64(np.stack(images))
print(images.shape)
ethnic = np.stack(ethnic)

# Normalize data
images /= 255.0
# Save to disk
f = h5py.File("images.h5", "w")
# Create images dataset
X_dset = f.create_dataset('data', images.shape, dtype='f')
X_dset[:] = images

# Create annotation
y_dset = f.create_dataset('ethnic', ethnic.shape, dtype='i')
y_dset[:] = ethnic

f.close()

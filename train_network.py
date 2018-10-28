import numpy as np
from keras.optimizers import Adam
from keras.engine import Model
from keras.layers import Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
import gc
import time
import h5py


def vgg_face_model(classes=4, hidden_dim=512, shape=(160, 160, 3)):
    # Convolution Features
    model = VGG16(include_top=False, input_shape=shape)
    last_layer = model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(hidden_dim, activation='relu', name='fc4096')(x)
    x = Dense(hidden_dim, activation='relu', name='fc4097')(x)
    out = Dense(classes, activation='softmax', name='fc1000')(x)
    vgg_model = Model(model.input, out)
    return vgg_model


# ==============================load data====================================

f = h5py.File('images.h5', 'r')
X_data = np.array(f['data'])
y_data = np.array(f['ethnic'])
y_data=to_categorical(y_data,num_classes=4)
X_train, y_train, X_dev, y_dev = train_test_split(X_data, y_data, test_size=0.3,random_state=87)
X_data=None
y_data=None
gc.collect()
# normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev = scaler.transform(X_dev)
with open("Scaler_parameters.txt", 'w') as f:
    print("mean is {}. variance is {}".format(scaler.mean_, scaler.var_))
    f.write("mean is {}. variance is {}".format(scaler.mean_, scaler.var_))

# =============================hypermeters====================================
learning_rate = 1e-3
optimizaion_method = "Adam"
batch_size = 320
iteration_times = 100
# hypermeter objects
adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.99)
# =============================build model====================================
model = vgg_face_model()
checkpoint = ModelCheckpoint('weights_ethnic.hdf5', monitor='val_acc', verbose=2, save_best_only=True,
                             save_weights_only=True, mode='max')
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=iteration_times,
          verbose=1,
          callbacks=[checkpoint,TensorBoard(log_dir='log')],
          validation_data=(X_dev, y_dev),
          initial_epoch=0)
localtime = time.strftime('%Y-%m-%d_%H%M%S', time.localtime(time.time()))
filename = localtime + '.h5'
model.save(filename)
print(model.summary())

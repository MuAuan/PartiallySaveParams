"""
Assuming the original model looks like this:
    model = Sequential()
    model.add(Dense(2, input_dim=3, name='dense_1'))
    model.add(Dense(3, name='dense_2'))
    ...
    model.save_weights(fname)


# new model
model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))  # will be loaded
model.add(Dense(10, name='new_dense'))  # will not be loaded

# load weights from first model; will only affect the first layer, dense_1.
model.load_weights(fname, by_name=True)
"""

from __future__ import print_function
import keras
from keras import layers, models, optimizers
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D,GlobalAveragePooling2D
from keras.layers import Input
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import ZeroPadding2D
from getDataSet import getDataSet
import numpy as np

batch_size = 5
num_classes = 3
epochs = 3
data_augmentation = True
img_rows,img_cols=300,300  #128,128       #224,224  #300,300

# The data, shuffled and split between train and test sets:
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, y_train, x_test, y_test = getDataSet(img_rows,img_cols)

#x_train=x_train[0:1000]
#y_train=y_train[0:1000]
#x_test=x_test[0:1000]
#y_test=y_test[0:1000]

#print('x_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

    

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#model = Sequential()

def model_cifar(input_shape, num_classes=10):
    x = Input(shape=input_shape)
    
    filter=64
    # Block 1
    conv1_1 = Conv2D(filter*1, (3, 3), name='conv1_1', padding='same', activation='relu')(x)
    conv1_2 = Conv2D(filter*1, (3, 3), name='conv1_2', padding='same', activation='relu')(conv1_1)
    conv1_2 = BatchNormalization(axis=3)(conv1_2)  
    drop_1 = Dropout(0.5)(conv1_2)               
    pool1 = AveragePooling2D(name='pool1', pool_size=(2, 2), strides=(2, 2), padding='same', )(conv1_2)

    # Block 2
    conv2_1 = Conv2D(filter*2, (3, 3), name='conv2_1', padding='same', activation='relu')(pool1)
    conv2_2 = Conv2D(filter*2, (3, 3), name='conv2_2', padding='same', activation='relu')(conv2_1)
    conv2_2 = BatchNormalization(axis=3)(conv2_2)  
    drop_2 = Dropout(0.5)(conv2_2)               
    pool2 = AveragePooling2D(name='pool2', pool_size=(2, 2), strides=(2, 2), padding='same')(conv2_2)

    # Block 3
    conv3_1 = Conv2D(filter*4, (3, 3), name='conv3_1', padding='same', activation='relu')(pool2)
    conv3_2 = Conv2D(filter*4, (3, 3), name='conv3_2', padding='same', activation='relu')(conv3_1)
    conv3_3 = Conv2D(filter*4, (3, 3), name='conv3_3', padding='same', activation='relu')(conv3_2)
    conv3_3 = BatchNormalization(axis=3)(conv3_3)  
    drop_3 = Dropout(0.5)(conv3_3)               
    pool3 = AveragePooling2D(name='pool3', pool_size=(2, 2), strides=(2, 2), padding='same')(conv3_3)
    
    # Block 4
    conv4_1 = Conv2D(filter*8, (3, 3), name='conv4_1', padding='same', activation='relu')(pool3)
    conv4_2 = Conv2D(filter*8, (3, 3), name='conv4_2', padding='same', activation='relu')(conv4_1)
    conv4_3 = Conv2D(filter*8, (3, 3), name='conv4_3', padding='same', activation='relu')(conv4_2)
    conv4_3 = BatchNormalization(axis=3)(conv4_3)  
    drop_4 = Dropout(0.5)(conv4_3)               
    pool4 = AveragePooling2D(name='pool4', pool_size=(2, 2), strides=(2, 2), padding='same')(conv4_3)
    
    # Block 5
    conv5_1 = Conv2D(filter*8, (3, 3), name='conv5_1', padding='same', activation='relu')(pool4)
    conv5_2 = Conv2D(filter*8, (3, 3), name='conv5_2', padding='same', activation='relu')(conv5_1)
    conv5_3 = Conv2D(filter*8, (3, 3), name='conv5_3', padding='same', activation='relu')(conv5_2)
    conv5_3 = BatchNormalization(axis=3)(conv5_3)  
    drop_5 = Dropout(0.5)(conv5_3)               
    pool5 = AveragePooling2D(name='pool5', pool_size=(3, 3), strides=(1, 1), padding='same')(conv5_3)
    """
    # FC6
    fc6 = Conv2D(filter*16, (3, 3), name='fc6', dilation_rate=(6, 6), padding='same', activation='relu')(pool5)  #pool5 16
    fc6 = Dropout(0.5, name='drop6')(fc6)

    # FC7
    fc7 = Conv2D(filter*16, (1, 1), name='fc7', padding='same', activation='relu')(fc6)  #16
    fc7 = Dropout(0.5, name='drop7')(fc7)

    # Block 6
    conv6_1 = Conv2D(filter*8, (1, 1), name='conv6_1', padding='same', activation='relu')(fc7) #8
    conv6_2 = Conv2D(filter*16, (3, 3), name='conv6_2', strides=(2, 2), padding='same', activation='relu')(conv6_1) #16
    conv6_2 = BatchNormalization(axis=3)(conv6_2)  
    conv6_2 = Dropout(0.5)(conv6_2)               

    # Block 7
    conv7_1 = Conv2D(filter*4, (1, 1), name='conv7_1', padding='same', activation='relu')(conv6_2) #4
    conv7_1z = ZeroPadding2D(name='conv7_1z')(conv7_1)
    conv7_2 = Conv2D(filter*4, (3, 3), name='conv7_2', padding='valid', strides=(2, 2), activation='relu')(conv7_1z) #4
    conv7_2 = BatchNormalization(axis=3)(conv7_2)  
    conv7_2 = Dropout(0.5)(conv7_2)               

    # Block 8
    conv8_1 = Conv2D(filter*2, (1, 1), name='conv8_1', padding='same', activation='relu')(conv7_2)  #2
    conv8_2 = Conv2D(filter*32, (3, 3), name='conv8_2', padding='same', strides=(2, 2), activation='relu')(conv8_1) #4
    conv8_2 = BatchNormalization(axis=3)(conv8_2)  
    conv8_2 = Dropout(0.5)(conv8_2)               

    # Last Pool
    #pool6 = GlobalAveragePooling2D(name='pool6')(conv8_2) #GlobalAveragePooling2D
    """
    flatten_1 = Flatten()(pool5)   #conv8_2)
    dense_1 = Dense(512, name='dense_1')(flatten_1)
    act_1 = Activation('relu')(dense_1)
    drop_6 = Dropout(0.5)(act_1)
    dense_2 = Dense(num_classes, name='dense_2')(drop_6)
    
    softmax = Activation('softmax')(dense_2)
    
    return models.Model(x, outputs=softmax)


model=model_cifar(input_shape=[img_rows,img_cols, 3], num_classes=3)
# load the weights from the last epoch
model.load_weights('weights_SSD300.hdf5', by_name=True)

freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
          'conv2_1', 'conv2_2', 'pool2',
          'conv3_1', 'conv3_2', 'conv3_3', 'pool3',
          'conv4_1', 'conv4_2', 'conv4_3', 'pool4',
          'conv5_1', 'conv5_2', 'conv5_3', 'pool5']
"""
for L in model.layers:
    if L.name in freeze:
        L.trainable = False
"""

def schedule(epoch, decay=0.8):   #0.9
    return base_lr * decay**(epoch)

base_lr = 0.0001
# initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=base_lr, decay=1e-6)
opt = keras.optimizers.Adam(lr=base_lr)

csv_logger = keras.callbacks.CSVLogger('./checkpoints/training.log', separator=',', append=True)
weights_save=keras.callbacks.ModelCheckpoint('./checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                             verbose=1,
                                             save_weights_only=True)
learnRateSchedule=keras.callbacks.LearningRateScheduler(schedule)

callbacks = [weights_save, csv_logger, learnRateSchedule]

       
# Let's train the model using opt
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])        
        
model.summary()        
        
# load weights for every block as the same name block
#model.load_weights("block_1_1_009", by_name=True)

#x_train = x_train.astype('float32')
x_train = np.array(x_train, dtype=np.float32)
#x_test = x_test.astype('float32')
x_test = np.array(x_test, dtype=np.float32)
x_train /= 255
x_test /= 255


if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    for j in range(10):
        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=(x_test, y_test))

        # save weights every epoch
        
        model.save_weights("all_block_{0:03d}".format(j))
        
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 32, 32, 3)         0
_________________________________________________________________
conv1_1 (Conv2D)             (None, 32, 32, 64)        1792
_________________________________________________________________
conv1_2 (Conv2D)             (None, 32, 32, 64)        36928
_________________________________________________________________
pool1 (AveragePooling2D)     (None, 16, 16, 64)        0
_________________________________________________________________
conv2_1 (Conv2D)             (None, 16, 16, 128)       73856
_________________________________________________________________
conv2_2 (Conv2D)             (None, 16, 16, 128)       147584
_________________________________________________________________
pool2 (AveragePooling2D)     (None, 8, 8, 128)         0
_________________________________________________________________
conv3_1 (Conv2D)             (None, 8, 8, 256)         295168
_________________________________________________________________
conv3_2 (Conv2D)             (None, 8, 8, 256)         590080
_________________________________________________________________
conv3_3 (Conv2D)             (None, 8, 8, 256)         590080
_________________________________________________________________
pool3 (AveragePooling2D)     (None, 4, 4, 256)         0
_________________________________________________________________
conv4_1 (Conv2D)             (None, 4, 4, 512)         1180160
_________________________________________________________________
conv4_2 (Conv2D)             (None, 4, 4, 512)         2359808
_________________________________________________________________
conv4_3 (Conv2D)             (None, 4, 4, 512)         2359808
_________________________________________________________________
pool4 (AveragePooling2D)     (None, 2, 2, 512)         0
_________________________________________________________________
conv5_1 (Conv2D)             (None, 2, 2, 512)         2359808
_________________________________________________________________
conv5_2 (Conv2D)             (None, 2, 2, 512)         2359808
_________________________________________________________________
conv5_3 (Conv2D)             (None, 2, 2, 512)         2359808
_________________________________________________________________
pool5 (AveragePooling2D)     (None, 2, 2, 512)         0
_________________________________________________________________
fc6 (Conv2D)                 (None, 2, 2, 1024)        4719616
_________________________________________________________________
drop6 (Dropout)              (None, 2, 2, 1024)        0
_________________________________________________________________
fc7 (Conv2D)                 (None, 2, 2, 1024)        1049600
_________________________________________________________________
drop7 (Dropout)              (None, 2, 2, 1024)        0
_________________________________________________________________
conv6_1 (Conv2D)             (None, 2, 2, 512)         524800
_________________________________________________________________
conv6_2 (Conv2D)             (None, 1, 1, 1024)        4719616
_________________________________________________________________
dropout_1 (Dropout)          (None, 1, 1, 1024)        0
_________________________________________________________________
conv7_1 (Conv2D)             (None, 1, 1, 256)         262400
_________________________________________________________________
conv7_1z (ZeroPadding2D)     (None, 3, 3, 256)         0
_________________________________________________________________
conv7_2 (Conv2D)             (None, 1, 1, 256)         590080
_________________________________________________________________
dropout_2 (Dropout)          (None, 1, 1, 256)         0
_________________________________________________________________
conv8_1 (Conv2D)             (None, 1, 1, 128)         32896
_________________________________________________________________
conv8_2 (Conv2D)             (None, 1, 1, 2048)        2361344
_________________________________________________________________
dropout_3 (Dropout)          (None, 1, 1, 2048)        0
_________________________________________________________________
flatten_1 (Flatten)          (None, 2048)              0
_________________________________________________________________
dense_1 (Dense)              (None, 512)               1049088
_________________________________________________________________
activation_1 (Activation)    (None, 512)               0
_________________________________________________________________
dropout_4 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 10)                5130
_________________________________________________________________
activation_2 (Activation)    (None, 10)                0
=================================================================
Total params: 30,029,258
Trainable params: 15,314,570
Non-trainable params: 14,714,688
_________________________________________________________________

Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 32, 32, 3)         0
_________________________________________________________________
conv1_1 (Conv2D)             (None, 32, 32, 64)        1792
_________________________________________________________________
conv1_2 (Conv2D)             (None, 32, 32, 64)        36928
_________________________________________________________________
dropout_1 (Dropout)          (None, 32, 32, 64)        0
_________________________________________________________________
pool1 (AveragePooling2D)     (None, 16, 16, 64)        0
_________________________________________________________________
conv2_1 (Conv2D)             (None, 16, 16, 128)       73856
_________________________________________________________________
conv2_2 (Conv2D)             (None, 16, 16, 128)       147584
_________________________________________________________________
dropout_2 (Dropout)          (None, 16, 16, 128)       0
_________________________________________________________________
pool2 (AveragePooling2D)     (None, 8, 8, 128)         0
_________________________________________________________________
conv3_1 (Conv2D)             (None, 8, 8, 256)         295168
_________________________________________________________________
conv3_2 (Conv2D)             (None, 8, 8, 256)         590080
_________________________________________________________________
conv3_3 (Conv2D)             (None, 8, 8, 256)         590080
_________________________________________________________________
dropout_3 (Dropout)          (None, 8, 8, 256)         0
_________________________________________________________________
pool3 (AveragePooling2D)     (None, 4, 4, 256)         0
_________________________________________________________________
conv4_1 (Conv2D)             (None, 4, 4, 512)         1180160
_________________________________________________________________
conv4_2 (Conv2D)             (None, 4, 4, 512)         2359808
_________________________________________________________________
conv4_3 (Conv2D)             (None, 4, 4, 512)         2359808
_________________________________________________________________
dropout_4 (Dropout)          (None, 4, 4, 512)         0
_________________________________________________________________
pool4 (AveragePooling2D)     (None, 2, 2, 512)         0
_________________________________________________________________
conv5_1 (Conv2D)             (None, 2, 2, 512)         2359808
_________________________________________________________________
conv5_2 (Conv2D)             (None, 2, 2, 512)         2359808
_________________________________________________________________
conv5_3 (Conv2D)             (None, 2, 2, 512)         2359808
_________________________________________________________________
dropout_5 (Dropout)          (None, 2, 2, 512)         0
_________________________________________________________________
pool5 (AveragePooling2D)     (None, 2, 2, 512)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 2048)              0
_________________________________________________________________
dense_1 (Dense)              (None, 512)               1049088
_________________________________________________________________
activation_1 (Activation)    (None, 512)               0
_________________________________________________________________
dropout_6 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 10)                5130
_________________________________________________________________
activation_2 (Activation)    (None, 10)                0
=================================================================
Total params: 15,768,906
Trainable params: 15,768,906
Non-trainable params: 0
_________________________________________________________________
"""
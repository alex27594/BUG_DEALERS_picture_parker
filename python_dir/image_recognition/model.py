from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

data_root = "/home/alexander/repos/tret_vk/data/"
train_path = data_root + "train"
test_path = data_root + "test"

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 224, 224), data_format="channels_first"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

model.add(Conv2D(32, (3, 3), data_format="channels_first"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

model.add(Conv2D(64, (3, 3), data_format="channels_first"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(23))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    data_format="channels_first"
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    data_format="channels_first"
)

batch_size = 230

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
early_stopping = EarlyStopping(monitor='val_acc', min_delta=5e-3, patience=3, mode='auto')
callbacks_list = [checkpoint, early_stopping]

model.fit_generator(
    train_generator,
    steps_per_epoch=10,
    epochs=40,
    validation_data=validation_generator,
    validation_steps=5,
    callbacks=callbacks_list
)
model.save_weights('first_try.h5')

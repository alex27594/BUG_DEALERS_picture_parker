from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3

data_root = "/home/alexander/repos/tret_vk/data/"
train_path = data_root + "all_train"
test_path = data_root + "all_test"


# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=(3, 224, 224), data_format="channels_first"))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
#
# model.add(Conv2D(32, (3, 3), data_format="channels_first"))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
#
# model.add(Conv2D(64, (3, 3), data_format="channels_first"))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
#
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(23))
# model.add(Activation('sigmoid'))
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(190, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(19, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

train_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

batch_size = 100

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

filepath="new-model/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max', period=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=5e-3, patience=5, mode='auto')
callbacks_list = [checkpoint, early_stopping]

model.fit_generator(
    train_generator,
    steps_per_epoch=10,
    epochs=17,
    validation_data=validation_generator,
    validation_steps=3,
    callbacks=callbacks_list
)

for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

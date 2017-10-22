import numpy as np
from keras.models import load_model
from scipy.misc import imread, imresize
from scipy.ndimage.filters import gaussian_filter


model = load_model('new-model/weights-improvement-10-0.19.hdf5')
print("----------")
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
# x = imread("/home/alexander/repos/tret_vk/data/real_test/8XniFIyujwg.jpg")
x = imread("/home/alexander/repos/tret_vk/data/all_original_pictures/Zaporozhci_pishut/Zaporozhci_pishut.jpg")
# x = imread("/home/alexander/repos/tret_vk/data/all_original_pictures/Christ/Christ.jpg")
# x = imread("/home/alexander/repos/tret_vk/data/original_pictures/girl_with_peaches.jpg")
# x = imread("/home/alexander/repos/tret_vk/data/original_pictures/demon.jpg")
x = gaussian_filter(x, (3, 3, 0))
x = imresize(x, (224, 224))
x = np.reshape(x, [1, 224, 224, 3])
x = x / 255

print(max(model.predict(x)[0].tolist()))
print(sorted(list(enumerate(model.predict(x)[0].tolist())), key=lambda item: item[1]))

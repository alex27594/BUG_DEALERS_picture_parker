import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from scipy.ndimage.filters import gaussian_filter

datagen = ImageDataGenerator(
    channel_shift_range=20,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.5,
    fill_mode='nearest',
    preprocessing_function=lambda item: gaussian_filter(item, sigma=(3, 3, 0)))

data_root = "/home/alexander/repos/tret_vk/data/"
# paths = [item for item in os.listdir(data_root + "all_original_pictures")]
paths = ["demon", "girl_with_peaches", "horse_rideress", "Naturmort", "neravny_brak",
         "panteleymon_healer", "Portrait_Lev", "Portrait_Pavl", "swan_princess",
         "Voin_v_Jaipure", "Zaporozhci_pishut"]
for path in paths:
    for img_path in [item[:-4] for item in os.listdir(data_root + "all_original_pictures/" + path)]:
        img = load_img(data_root + "all_original_pictures/" + path + "/" + img_path + ".jpg")
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=data_root + "all_test/" + path,
                                  save_prefix=path,
                                  save_format='jpg'):
            i += 1
            if i > 30:
                break

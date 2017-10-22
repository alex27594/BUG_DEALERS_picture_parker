import warnings

import numpy as np
from keras.models import load_model
from scipy.misc import imread, imresize
from scipy.ndimage.filters import gaussian_filter
from flask import Flask
from flask import request

THRESHOLD = 0.4

inner_outer_link = {10: 8,
                    11: 10,
                    0: 1,
                    1: 2,
                    2: 3,
                    12: 11,
                    13: 12,
                    14: 13,
                    15: 14,
                    3: 5,
                    16: 6,
                    17: 15,
                    5: 18,
                    6: 19,
                    18: 16,
                    7: 21,
                    8: 22,
                    4: 7,
                    9: 23
                    }
model = load_model('weights-improvement-10-0.19.hdf5')
app = Flask(__name__)


@app.route('/', methods=['GET'])
def predict():
    path = request.args.get('path')
    x = imread(path)
    x = gaussian_filter(x, (3, 3, 0))
    x = imresize(x, (224, 224))
    x = np.reshape(x, [1, 224, 224, 3])
    x = x / 255
    if max(model.predict(x)[0].tolist()) < THRESHOLD:
        return str(-1)
    else:
        return str(inner_outer_link[sorted(list(enumerate(model.predict(x)[0].tolist())), key=lambda item: item[1])[-1][0]])

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        app.run()


import wave 
import numpy as np
import pandas as pd
from para import *
import librosa 
import matplotlib.pyplot as plt
import librosa.display
import scipy

from utils import *

dm = DataManager()
dm.add_rawData('../data/train.csv', mode='semi')
dm.preprocess()
dm.dump_data(train_spectrum_dir)
dm.load_data(train_spectrum_dir, mode='semi')

train_data = dm.get_data('train_data')[0]
semi_data = dm.get_data('semi_data')[0]
D_1 = train_data[2]
D_2 = train_data[6]


#D_1 = librosa.amplitude_to_db(D_1, ref=np.max)
#D_2 = librosa.amplitude_to_db(D_2, ref=np.max)

orin_shape_1 = D_1.shape
#D_1 = scipy.misc.imresize((D_1*255/(-80.0)), (fre_bin, 500), interp='bicubic') / 255 * (-80.0)

plt.subplot(1, 2, 1)
librosa.display.specshow(D_1, y_axis='log', x_axis='time', sr=sr)
plt.title('saxphone labled with shape {}'.format(orin_shape_1))
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()


orin_shape_2 = D_2.shape
#D_2 = scipy.misc.imresize((D_2*255/(-80.0)), (fre_bin, 500), interp='bicubic') / 255 * (-80.0)

plt.subplot(1, 2, 2)
librosa.display.specshow(D_2, y_axis='log', x_axis='time', sr=sr)
plt.title('cello labled with shape {}'.format(orin_shape_2))
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()



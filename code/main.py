from importlib import reload

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import pdb
import pandas as pd
import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, SpatialDropout1D
from keras.layers import LSTM  # new!
from keras.callbacks import ModelCheckpoint
import os
import sys
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import uol_redacoes_xml
import re

print(np.__version__)

reload(sys)
print(sys.getdefaultencoding())
# %matplotlib inline
PATH = '../Resources'
FILE_NAMES = os.listdir(PATH)


def main():
    essays = uol_redacoes_xml.load()
    print(len(essays))
    # ~2000
    print(essays[0].text)
    print(essays[3].criteria_scores)

    # texto original da primeira redação
    print([attr for attr in essays[0].__dir__() if not attr.startswith('_')])
    # exibe os atributos do objeto de redação (exceto os privados, que começam com '_')


if __name__ == '__main__':
    main()

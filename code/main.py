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

print(np.__version__)

reload(sys)
print(sys.getdefaultencoding())
# %matplotlib inline
PATH = '../Resources'
FILE_NAMES = os.listdir(PATH)


def load_data():
    for archive in os.listdir(PATH):
        data = pd.read_csv(os.path.join(PATH, archive), sep='\t', encoding='latin-1')
        print(data['essay'][0])


def training():
    # output directory name:
    output_dir = os.path.join(PATH, 'model_output/vanillaLSTM')

    # training:
    epochs = 4
    batch_size = 128

    # vector-space embedding:
    n_dim = 64
    n_unique_words = 10000
    max_review_length = 100  # lowered due to vanishing gradient over time
    pad_type = trunc_type = 'pre'
    drop_embed = 0.2

    # LSTM layer architecture:
    n_lstm = 256
    drop_lstm = 0.2

    # dense layer architecture:
    # n_dense = 256
    # dropout = 0.2

    (x_train, y_train), (x_valid, y_valid) = imdb.load_data(num_words=n_unique_words)  # removed n_words_to_skip

    x_train = pad_sequences(x_train, maxlen=max_review_length, padding=pad_type, truncating=trunc_type, value=0)
    x_valid = pad_sequences(x_valid, maxlen=max_review_length, padding=pad_type, truncating=trunc_type, value=0)

    model = Sequential()
    model.add(Embedding(n_unique_words, n_dim, input_length=max_review_length))
    model.add(SpatialDropout1D(drop_embed))
    model.add(LSTM(n_lstm, dropout=drop_lstm))
    # model.add(Dense(n_dense, activation='relu')) # typically don't see top dense layer in NLP like in
    # model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    modelcheckpoint = ModelCheckpoint(filepath=output_dir + "/weights.{epoch:02d}.hdf5")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # go have a gander at nvidia-smi
    # 85.2% validation accuracy in epoch 2
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_valid, y_valid),
              callbacks=[modelcheckpoint])


def labeler(example, index):
    return example, tf.cast(index, tf.int64)


def main():
    labeled_data_sets = []

    for i, archive in enumerate(FILE_NAMES):
        lines_dataset = tf.data.TextLineDataset(os.path.join(PATH, archive))
        labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
        labeled_data_sets.append(labeled_dataset)

    print(labeled_data_sets)
    BUFFER_SIZE = 50000
    BATCH_SIZE = 64
    TAKE_SIZE = 5000

    all_labeled_data = labeled_data_sets[0]
    for labeled_dataset in labeled_data_sets[1:]:
        all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

    all_labeled_data = all_labeled_data.shuffle(
        BUFFER_SIZE, reshuffle_each_iteration=False)

    for ex in all_labeled_data.take(5):
        print(ex)

    tokenizer = tfds.features.text.Tokenizer()

    vocabulary_set = set()
    for text_tensor, _ in all_labeled_data:
        some_tokens = tokenizer.tokenize(text_tensor.numpy())
        vocabulary_set.update(some_tokens)

    vocab_size = len(vocabulary_set)
    print(vocab_size)
    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
    example_text = next(iter(all_labeled_data))[0].numpy()
    print(example_text)
    encoded_example = encoder.encode(example_text)
    print(encoded_example)


if __name__ == '__main__':
    main()

import tensorflow as tf
import numpy as np
import os
import pdb
import pandas as pd
PATH = '../Resources'


def embed():
    inputs = np.random.random([32, 10, 8]).astype(np.float32)
    lstm = tf.keras.layers.LSTM(4)

    output = lstm(inputs)  # The output has shape `[32, 4]`.

    lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True)

    # whole_sequence_output has shape `[32, 10, 4]`.
    # final_memory_state and final_carry_state both have shape `[32, 4]`.

    whole_sequence_output, final_memory_state, final_carry_state = lstm(inputs)
    print(final_memory_state)


def main():
    for archive in os.listdir(PATH):
        data = pd.read_csv(os.path.join(PATH, archive), sep='\t', encoding='latin-1')
        print(data['essay'][0])


if __name__ == '__main__':
    main()

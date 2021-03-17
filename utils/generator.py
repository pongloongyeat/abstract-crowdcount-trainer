import random
import numpy as np
from tqdm import tqdm


"""
Data generator with custom input and output loading functions.
There are separate Train and Test generators for now since Keras
requires generators to have a while loop but we need the generator
to have a for loop so it can end and the MAE and MSE scores can
be computed.

This generator assumes your data is similar in naming
i.e. each file should be named

list_of_input_paths = ['a_1.jpg', 'a_2.jpg', 'a_4.jpg']

and not

list_of_input_paths = ['a_1.jpg', 'zzz_2.jpg', '9a8s8d_4.jpg']
"""
class CustomGenerator():
    def __init__(self, list_of_input_paths, list_of_output_paths,
                 get_input_func, get_output_func):
        self.list_of_input_paths = sorted(list_of_input_paths)
        self.list_of_output_paths = sorted(list_of_output_paths)
        self.get_input_func = get_input_func
        self.get_output_func = get_output_func
        self.steps_per_epoch = len(list_of_input_paths)

        if len(self.list_of_input_paths) != len(self.list_of_output_paths): ValueError("Different number of input and output files.")

    def train_generator(self):
        # Shuffle data
        indices = [x for x in range(len(self.list_of_input_paths))]
        random.shuffle(indices)

        index = 0
        while True:
            if index >= self.steps_per_epoch: index = 0

            input_data = np.asarray([self.get_input_func(self.list_of_input_paths[index])])
            output_data = np.asarray([self.get_output_func(self.list_of_output_paths[index])])

            index += 1

            yield input_data, output_data

    def test_generator(self):
        # Shuffle data
        indices = [x for x in range(len(self.list_of_input_paths))]
        random.shuffle(indices)

        for index in indices:
            input_data = np.asarray([self.get_input_func(self.list_of_input_paths[index])])
            output_data = np.asarray([self.get_output_func(self.list_of_output_paths[index])])

            index += 1

            yield input_data, output_data

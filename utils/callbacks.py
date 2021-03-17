import cv2
import keras
import os
import numpy as np
from tqdm import tqdm


class CalculateScore(keras.callbacks.Callback):
    """
    Calculates the MAE and MSE score since for our application,
    MAE and MSE calculation involves summing the points in the data
    before any operation.
    """

    def __init__(self, generator):
        self.generator = generator

    def on_epoch_end(self, epoch, logs=None):
        dataset_path = "data/ShanghaiTech/part_B"
        test_gen = self.generator.test_generator()

        y_true = []
        y_pred = []

        pbar = tqdm(test_gen)
        pbar.desc = "Validating..."

        for (x, y) in pbar:
            y_true.append(np.sum(y))
            y_pred.append(np.sum(self.model.predict(x)))

        # Convert to numpy array for operations
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        mae = np.mean(abs(y_true - y_pred))
        mse = np.sqrt(np.mean(np.square(y_true - y_pred)))

        print("\nEpoch {} - MAE: {}, MSE: {}".format(epoch + 1, np.round(mae, 4), np.round(mse, 4)))

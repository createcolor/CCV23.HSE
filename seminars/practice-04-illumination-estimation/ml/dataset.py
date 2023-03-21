from tensorflow import keras
import numpy as np
import pandas as pd
import os
import cv2


class CubePPLoader(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self,
            root_dir, 
            csv_file,
        ):
        self.images_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir,
                                self.images_frame.iloc[idx, 0])
        image = cv2.imread(img_name + '.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float64)
        image = image / 255.0
        image = cv2.resize(image, (96, 96))
        image = np.expand_dims(image, axis=0)
        light_source = self.images_frame.iloc[idx, 1:]
            
        return image, np.expand_dims(light_source.to_numpy(dtype=np.float64), axis=0)

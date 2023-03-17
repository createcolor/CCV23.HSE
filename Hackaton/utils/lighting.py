import numpy as np
import pandas as pd
from pathlib import Path
import os.path as osp


class Lighting:
    def __init__(self, seed=None):
        if seed is None:
            np.random.seed(42)
        else:
            np.random.seed(seed)

        dir = osp.join(Path(__file__).parent.absolute(), "./light/CIE_015_4_Data.xlsx")
        self.df = pd.read_excel(dir, sheet_name="SPD CIE illuminants")
        self.df = self.df.iloc[:, [0, 1, 4, 5, 6]]
        self.df.columns = self.df.iloc[1]
        self.df.columns.values[0], self.df.columns.values[1] = "nm", "A"
        self.df = self.df.iloc[2:, ...]

    def select_random_light(self):
        x1 = np.array(self.df["nm"]).astype(np.float64)
        x2 = np.linspace(400, 730, 34).astype(np.float64)
        select_random_light = np.random.choice(self.df.columns[1:])
        y1 = np.array(self.df[select_random_light]).astype(np.float64)
        y2 = np.interp(x2, x1, y1)

        return {select_random_light: y2}

import numpy as np
import h5py
import colour
import pandas as pd
from rich.progress import track


class Spectral2XYZ:
    def __init__(self, path_to_cmfs=None):
        if path_to_cmfs is None:
            cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
            self.cmfs = {
                "wavelengths": cmfs.wavelengths,
                "values": cmfs.values
            }
        else:
            self.cmfs = {
                'wavelengths': np.linspace(400, 730, 34),
                'values': []
            }

            df = pd.read_csv(path_to_cmfs)

            assert "wavelength" in df.columns, "you must have wavelength columns"
            assert "r" in df.columns, "you must have r columns"
            assert "g" in df.columns, "you must have g columns"
            assert "b" in df.columns, "you must have b columns"

            x1 = df["wavelength"]
            x2 = self.cmfs["wavelengths"]

            for ch in ["r", "g", "b"]:
                y1 = np.array(df[ch]).astype(np.float32)
                self.cmfs['values'].append(np.interp(x2, x1, y1))

            self.cmfs['values'] = np.array(self.cmfs['values']).T

    def _spectral_to_XYZ(self, spectral, light=None):
        sd = dict(zip(np.linspace(400, 730, 34), spectral))
        if light is not None:
            light_source = dict(zip(np.linspace(400, 730, 34), list(light.values())[0]))
            sd = {nm: sd[nm] * light_source[nm] for nm in light_source}
            sd = {nm: sd[nm] / max(sd.values()) for nm in light_source}

        wavelengths = np.array(list(sd.keys()))
        power = np.array(list(sd.values())).reshape(-1, 1)

        mask_for_sd = np.isin(wavelengths, self.cmfs["wavelengths"])
        mask_for_cmfs = np.isin(self.cmfs["wavelengths"], wavelengths)

        power = power[mask_for_sd]
        cmfs_values = self.cmfs['values'][mask_for_cmfs]
        cmfs_values /= cmfs_values[:, 1].sum()
        xyz = (power * cmfs_values).sum(axis=0)
        return xyz

    def spectral_to_XYZ(self, spectral_img, light=None):
        xyz_img = np.zeros_like(spectral_img, shape=(3, 512, 512)).astype(np.float64)
        for y in track(range(spectral_img.shape[1]), description="Processing..."):
            for x in range(spectral_img.shape[2]):
                xyz = self._spectral_to_XYZ(spectral_img[:, y, x], light)
                xyz_img[:, y, x] = xyz

        return xyz_img

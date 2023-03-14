import numpy as np
import warnings


class XYZ_TO_SRGB:
    def __init__(self, LIN_RGB_MATRIX=None):
        if LIN_RGB_MATRIX is not None:
            self.LIN_RGB_MATRIX = LIN_RGB_MATRIX
        else:
            self.LIN_RGB_MATRIX = np.array(
                [
                    [3.2404542, -1.5371385, -0.4985314],
                    [-0.9692660, 1.8760108, 0.0415560],
                    [0.0556434, -0.2040259, 1.0572252],
                ]
            )

    def _XYZ_to_linRGB(self, xyz):
        xyz = np.array(xyz)
        return xyz @ self.LIN_RGB_MATRIX.T

    def _linRGB2sRGB(self, img):
        """
        img: float32 image [0, 1]
        """
        if img.dtype != np.float32:
            warnings.warn("ffs, use float32 not %s" % img.dtype)
        if img.min() < 0 or img.max() > 1:
            warnings.warn(
                "ffs, the range should be in [0, 1] not [%f %f]" % (img.min(), img.max())
            )

        thres = 0.0031308
        a = 0.055

        img = np.clip(img, 0, 1)

        for y in range(0, img.shape[0], 64):
            for x in range(0, img.shape[1], 64):
                s = slice(y, y + 64), slice(x, x + 64)
                fragment = img[s]
                low = fragment <= thres

                fragment[low] *= 12.92
                fragment[~low] = (1 + a) * fragment[~low] ** (1 / 2.4) - a

        return img

    def XYZ_to_sRGB(self, xyz):
        lin = self._XYZ_to_linRGB(xyz)
        return self._linRGB2sRGB(lin)

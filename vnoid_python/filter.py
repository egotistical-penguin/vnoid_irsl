import math

import numpy as np


class Filter:
    """Python port of vnoid/src/filter.{h,cpp}."""

    class Type:
        NoneType = 0
        FirstOrderLPF = 1
        FirstOrderLPF2 = 2
        Butterworth3 = 3

    def __init__(self, cutoff=1.0, filter_type=None, yd_max=0.0):
        self.type = (self.Type.FirstOrderLPF
                     if filter_type is None else filter_type)
        self.u = 0.0
        self.ydd = 0.0
        self.yd = 0.0
        self.y = 0.0
        self.yd_max = float(yd_max)
        self.first = True
        self.SetCutoff(cutoff)

    def SetCutoff(self, cutoff):
        self.w = 2.0 * math.pi * float(cutoff)
        self.w2 = self.w * self.w
        self.w3 = self.w * self.w2

    def reset(self, value=0.0):
        value = np.asarray(value, dtype=float)
        self.u = value.copy()
        self.y = value.copy()
        self.yd = np.zeros_like(value)
        self.ydd = np.zeros_like(value)
        self.first = True

    @staticmethod
    def _return_value(value):
        value = np.asarray(value)
        if value.ndim == 0:
            return float(value)
        return value.copy()

    def __call__(self, value, dt):
        value = np.asarray(value, dtype=float)
        if self.first:
            self.y = value.copy()
            self.yd = np.zeros_like(value)
            self.ydd = np.zeros_like(value)
            self.first = False

        self.u = value.copy()
        dt = float(dt)

        if self.type == self.Type.NoneType:
            self.y = self.u.copy()
        elif self.type == self.Type.FirstOrderLPF:
            self.yd = -self.w * (self.y - self.u)
            if self.yd_max > 0.0:
                self.yd = np.clip(self.yd, -self.yd_max, self.yd_max)
            self.y = self.y + self.yd * dt
        elif self.type == self.Type.FirstOrderLPF2:
            alpha = math.exp(-self.w * dt)
            self.y = alpha * self.y + (1.0 - alpha) * self.u
        elif self.type == self.Type.Butterworth3:
            self.y = self.y + self.yd * dt
            self.yd = self.yd + self.ydd * dt
            if self.yd_max > 0.0:
                self.yd = np.clip(self.yd, -self.yd_max, self.yd_max)
            self.ydd = self.ydd + (
                -2.0 * self.w * self.ydd
                -2.0 * self.w2 * self.yd
                -self.w3 * self.y
                +self.w3 * self.u) * dt
        else:
            raise ValueError('unknown filter type: {}'.format(self.type))

        return self._return_value(self.y)

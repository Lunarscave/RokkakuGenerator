import math
import random

import numpy as np
from numpy import ndarray

from utils import assert_util, point_util


class Strokes:
    """
    Meta data for strokes
    """
    def __init__(self) -> None:
        """
        Init strokes and set value to None
        """
        self.value = None
        self.nums = 0

    def __sort_value__(self) -> None:
        """
         Sort strokes value by stroke index.
        """
        value = list(self.value)
        value.sort(key=lambda p: p[-1])
        self.value = np.array(value)

    @classmethod
    def __validate_strokes__(cls, value):
        assert_util.is_true(value.ndim == 2 and value.shape[1] == 4, "Strokes value invalid.")

    @classmethod
    def __validate_points__(cls, points):
        assert_util.is_true(points.ndim == 2 and points.shape[1] == 3, "Points value invalid.")

    def set_value(self, value: ndarray):
        """
        Set strokes value
        """
        self.value = value
        self.__sort_value__()
        self.nums = self.value[-1][-1] + 1
        return self

    def get_value(self):
        return self.value

    def append_strokes(self, *values: ndarray):
        """
        Append (tuple of) strokes value
        """
        for value in values:
            self.__validate_strokes__(value)

        for value in values:
            cur_strokes = value[0][-1]
            for i, _ in enumerate(value):
                if value[i][-1] != cur_strokes:
                    self.nums += 1
                    cur_strokes = value[i][-1]
                value[i][-1] = self.nums
            self.value = point_util.concatenate_points(self.value, value)

        self.__sort_value__()
        return self

    def append_points(self, *points_arr: ndarray):
        """
        Append (tuple of) into points
        """
        for points in points_arr:
            self.__validate_points__(points)

        for points in points_arr:
            value = np.insert(points, 3, [self.nums for _ in points], axis=1)
            self.nums += 1
            self.value = point_util.concatenate_points(self.value, value)

        self.__sort_value__()
        return self

    @classmethod
    def load_strokes(cls, *values: ndarray):
        """
        Load (tuple of) strokes value
        """
        strokes = Strokes()
        strokes.set_value(values[0])
        strokes.append_strokes(*values[1:])
        return strokes

    @classmethod
    def load_points(cls, *points: ndarray):
        """
        Load strokes value
        """
        strokes = Strokes()
        strokes.set_value(cls.mark_points3d(points[0], 0))
        strokes.append_points(*points[1:])
        return strokes

    @classmethod
    def mark_points3d(cls,
                      points: ndarray,
                      tag: int) -> ndarray:
        """
        Mark points at the 4th index
        """

        return np.insert(points, 3, [tag for _ in points], axis=1)

    def rotate_points3d(self,
                        radian: float = None,
                        radian_range: tuple = None,
                        degree_range: tuple = (0, 0),
                        base: ndarray = np.array([0, 0, 0]),
                        axis: int = 2):
        """
        Rotate pointset around axis(x = 0, y = 1, z = 2) in order x, y, z
        """

        def rotate_point3d(point: ndarray,
                           rad: float = None,
                           rad_range: tuple = (0, 0),
                           base_point: ndarray = np.array([0, 0, 0]),
                           axis: int = 2) -> ndarray:
            """
            Rotate point around axis(x = 0, y = 1, z = 2) in order x, y, z
            """
            rad = random.uniform(*rad_range) if rad is None else rad
            flags = np.array([i != axis for i in range(0, 3)])
            rotate_matrix = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])
            rotate_point = point[0: 3][flags]
            base_point = base_point[flags]

            point[0: 3][flags] = np.matmul(rotate_matrix, rotate_point - base_point) + base_point
            return point

        strokes = self.value
        radian_range = tuple(np.deg2rad(degree_range)) if radian_range is None else radian_range
        radian = random.uniform(*radian_range) if radian is None else radian
        self.value = np.array([rotate_point3d(point, rad=radian, base_point=base, axis=axis) for point in strokes])
        return self

    def move_points3d(self,
                      vector: ndarray = None,
                      dx_range: tuple = (0, 0),
                      dy_range: tuple = (0, 0),
                      dz_range: tuple = (0, 0),
                      dx: float = 0,
                      dy: float = 0,
                      dz: float = 0):
        """
        Move points with vector or dx(y/z)
        """

        def move_point3d(point: ndarray,
                         vector: ndarray = None,
                         dx: float = 0,
                         dy: float = 0,
                         dz: float = 0) -> ndarray:
            """
            Move point with vector or dx(y/z)
            """
            vector = np.array([dx, dz, dy]) if vector is None else vector
            point[0: 3] = point[0: 3] + vector
            return point

        dx = random.uniform(*dx_range) if vector is None else dx
        dy = random.uniform(*dy_range) if vector is None else dy
        dz = random.uniform(*dz_range) if vector is None else dz
        self.value = np.array([move_point3d(point, vector, dx, dy, dz) for point in self.value])
        return self

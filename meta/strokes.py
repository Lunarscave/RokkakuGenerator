import math
import random
from typing import Any

import numpy as np
from numpy import ndarray, dtype
from typing_extensions import Self

from utils import assert_util, generate_point_util


class Strokes:
    """
    Meta data for strokes.
    """
    def __init__(self) -> None:
        """
        Init strokes and set value to None.
        """
        self.value: ndarray[ndarray] | None = None
        self.nums: int = 0

    def __sort_value__(self) -> None:
        """
        Sort strokes value by stroke index.
        """
        value = list(self.value)
        value.sort(key=lambda p: p[-1])
        self.value = np.array(value)

    @classmethod
    def __validate_strokes__(
            cls,
            value: ndarray[ndarray]
    ) -> None:
        """
        Validate strokes value.
        :param value: Strokes value.
        """
        assert_util.is_true(value.ndim == 2 and value.shape[1] == 4, "Strokes value invalid.")

    @classmethod
    def __validate_points__(
            cls,
            points: ndarray[ndarray]
    ) -> None:
        """
        Validate points value.
        :param points: Strokes points.
        """
        assert_util.is_true(points.ndim == 2 and points.shape[1] == 3, "Points value invalid.")

    def set_value(
            self,
            value: ndarray[ndarray]
    ) -> Self:
        """
        Set strokes value.
        :param value: Strokes value.
        :return: Stroke.
        """
        self.value = value
        self.__sort_value__()
        self.nums = self.value[-1][-1]
        return self

    def get_value(self) -> ndarray[ndarray]:
        """
        Get strokes value.
        :return: Strokes value.
        """
        return self.value

    def append_strokes(
            self,
            *values: ndarray[ndarray]
    ) -> Self:
        """
        Append (tuple of) strokes value.
        :param values: Strokes values.
        :return: Strokes.
        """
        for value in values:
            self.__validate_strokes__(value)

        self.nums += 1
        for value in values:
            cur_strokes = value[0][-1]
            for point in value:
                if point[-1] != cur_strokes:
                    self.nums += 1
                    cur_strokes = point[-1]
                point[-1] = self.nums
            self.value = generate_point_util.concatenate_points(self.value, value)

        self.__sort_value__()
        return self

    def append_points(
            self,
            *points_arr: ndarray[ndarray]
    ) -> Self:
        """
        Append (tuple of) into points.
        :param points_arr: Points value.
        :return: Strokes.
        """
        for points in points_arr:
            self.__validate_points__(points)

        for points in points_arr:
            self.nums += 1
            value = np.insert(points, 3, [self.nums for _ in points], axis=1)
            self.value = generate_point_util.concatenate_points(self.value, value)

        self.__sort_value__()
        return self

    @classmethod
    def load_strokes(
            cls,
            *values: ndarray[ndarray]
    ) -> Self:
        """
        Load (tuple of) strokes value.
        :param values: Strokes values.
        :return: Strokes.
        """
        strokes = Strokes()
        strokes.set_value(values[0])
        strokes.append_strokes(*values[1:])
        return strokes

    @classmethod
    def load_points(
            cls,
            *points: ndarray[ndarray] | ndarray
    ) -> Self:
        """
        Load strokes value by points.
        :param points: Strokes points.
        :return: Strokes.
        """
        strokes = Strokes()
        strokes.set_value(cls.mark_points3d(points[0], 1))
        strokes.append_points(*points[1:])
        return strokes

    @classmethod
    def mark_points3d(
            cls,
            points: ndarray[ndarray],
            tag: int
    ) -> ndarray[ndarray] | None:
        """
        Mark points at the 4th index.
        :param points: Points value.
        :param tag: Stroke tag (which stroke).
        :return: Marked strokes value.
        """
        stroke_points = [np.array(list(point) + [tag]) for point in points]
        return np.array(stroke_points)

    def rotate_points3d(
            self,
            radian: float = None,
            radian_range: tuple = None,
            degree_range: tuple = (0, 0),
            base: ndarray[ndarray] = np.array([0, 0, 0]),
            axis: int = 2
    ) -> Self:
        """
        Rotate pointset around axis(x = 0, y = 1, z = 2) in order x, y, z.
        :param radian: Rotate radian.
        :param radian_range: Rotate radian range.
        :param degree_range: Rotate degree range (convert to radian).
        :param base: Base point.
        :param axis: Rotate axis.
        :return: Rotated strokes.
        """
        def rotate_point3d(
                point: ndarray[ndarray],
                rad: float = None,
                rad_range: tuple = (0, 0),
                base_point: ndarray[ndarray] = np.array([0, 0, 0]),
                axis: int = 2
        ) -> ndarray[ndarray]:
            """
            Rotate point around axis(x = 0, y = 1, z = 2) in order x, y, z.
            :param point: Rotate point.
            :param rad: Rotate radian.
            :param rad_range: Rotate radian range.
            :param base_point: Base point.
            :param axis: Rotate axis.
            :return: Rotated point.
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

    def move_points3d(
            self,
            vector: ndarray[ndarray] = None,
            dx_range: tuple = (0, 0),
            dy_range: tuple = (0, 0),
            dz_range: tuple = (0, 0),
            dx: float = 0,
            dy: float = 0,
            dz: float = 0
    ) -> Self:
        """
        Move points with vector or dx(y/z).
        :param vector: Move vector.
        :param dx_range: Move x range.
        :param dy_range: Move y range.
        :param dz_range: Move z range.
        :param dx: Move x range.
        :param dy: Move y range.
        :param dz: Move z range.
        :return: Moved Strokes.
        """
        def move_point3d(
                point: ndarray[ndarray],
                vector: ndarray[ndarray] = None,
                dx: float = 0,
                dy: float = 0,
                dz: float = 0
        ) -> ndarray[ndarray]:
            """
            Move point with vector or dx(y/z)
            :param point: Point be moved.
            :param vector: Move vector.
            :param dx: Move dx.
            :param dy: Move dy.
            :param dz: Move dz.
            :return: Moved point.
            """
            vector = np.array([dx, dz, dy]) if vector is None else vector
            point[0: 3] = point[0: 3] + vector
            return point

        dx = random.uniform(*dx_range) if vector is None else dx
        dy = random.uniform(*dy_range) if vector is None else dy
        dz = random.uniform(*dz_range) if vector is None else dz
        self.value = np.array([move_point3d(point, vector, dx, dy, dz) for point in self.value])
        return self

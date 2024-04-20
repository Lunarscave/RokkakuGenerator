import random
from typing import Tuple

import numpy as np
from numpy import ndarray

from geometry import BaseGeometryHandler
from meta import Strokes, BuiltinGeometry
from utils import generate_point_util


class ShedHandler(BaseGeometryHandler):
    """
    Shed handler
    """

    def __init__(self):
        """
        Init shed handler with rules.
        """
        super().__init__()
        self.point_x_range: Tuple[float, float] | None = None
        self.th_degree_range: Tuple[float, float] | None = None
        self.r_range: Tuple[float, float] | None = None
        self.bottom_y_range: Tuple[float, float] | None = None
        self.arc_rot_degree_range: Tuple[float, float] | None = None
        self.endpoint_dithering: Tuple[float, float] | None = None

    def load_config(self) -> None:
        """
        Load shed handler config.
        """
        self.load_base_config()
        self.point_x_range = self._config_["point_x_range"]
        self.th_degree_range = self._config_["th_degree_range"]
        self.r_range = self._config_["r_range"]
        self.bottom_y_range = self._config_["bottom_y_range"]
        self.arc_rot_degree_range = self._config_["arc_rot_degree_range"]
        self.endpoint_dithering = self._config_["endpoint_dithering"]

    def prototype(self) -> BuiltinGeometry:
        """
        Get shed handler name.
        """
        return BuiltinGeometry.Shed

    def generate_strokes(self) -> Strokes:
        """
        Generate shed strokes.
        :return: Shed strokes.
        """

        # generate
        base = generate_point_util.get_point3d(x_range=self.point_x_range, y=0, z=0, dithering=self.point_dithering)
        r = random.uniform(*self.r_range)
        th = np.deg2rad(random.uniform(*self.th_degree_range))
        arc1 = generate_point_util.get_vertical_arc(
            density=self.density,
            r=r, th=th, y_range=self.bottom_y_range,
            rot_degree_range=self.arc_rot_degree_range,
            base=base, point_dithering=self.point_dithering
        )
        arc2 = generate_point_util.get_vertical_arc(
            density=self.density,
            r=r,
            th=th,
            y_range=self.bottom_y_range,
            rot_degree_range=self.arc_rot_degree_range,
            base=-base,
            point_dithering=self.point_dithering
        )

        line = generate_point_util.get_connect_line3d(
            lines=(arc1, arc2),
            density=self.density,
            equinox_range=self.equinox_range,
            endpoint_dithering=self.endpoint_dithering,
            point_dithering=self.point_dithering
        )

        # transform
        return Strokes.load_points(
            arc1, arc2, line
        ).rotate_points3d(
            degree_range=self.plane_rotate_degree_range
        ).move_points3d(
            dx_range=self.move_plane_range,
            dz_range=self.move_plane_range,
            dy_range=self.move_y_range
        )

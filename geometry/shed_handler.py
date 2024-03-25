import random

import numpy as np
from numpy import ndarray

from geometry import BaseGeometryHandler
from meta import Strokes
from utils import point_util


class ShedHandler(BaseGeometryHandler):
    """
    Shed handler
    """

    def __init__(self):
        super().__init__()
        self.point_x_range = None
        self.th_degree_range = None
        self.r_range = None
        self.bottom_y_range = None
        self.arc_rot_degree_range = None
        self.endpoint_dithering = None

    def load_config(self) -> None:
        self.load_base_config()
        self.point_x_range = self._config_["point_x_range"]
        self.th_degree_range = self._config_["th_degree_range"]
        self.r_range = self._config_["r_range"]
        self.bottom_y_range = self._config_["bottom_y_range"]
        self.arc_rot_degree_range = self._config_["arc_rot_degree_range"]
        self.endpoint_dithering = self._config_["endpoint_dithering"]

    def name(self) -> str:
        return "shed"

    def generate_strokes(self) -> ndarray:
        # generate
        base = point_util.get_point3d(x_range=self.point_x_range, y=0, z=0, dithering=self.point_dithering)
        r = random.uniform(*self.r_range)
        th = np.deg2rad(random.uniform(*self.th_degree_range))
        arc1 = point_util.get_vertical_arc(density=self.density,
                                           r=r,
                                           th=th,
                                           y_range=self.bottom_y_range,
                                           rot_degree_range=self.arc_rot_degree_range,
                                           base=base,
                                           point_dithering=self.point_dithering)
        arc2 = point_util.get_vertical_arc(density=self.density,
                                           r=r,
                                           th=th,
                                           y_range=self.bottom_y_range,
                                           rot_degree_range=self.arc_rot_degree_range,
                                           base=-base,
                                           point_dithering=self.point_dithering)

        line = point_util.get_connect_line3d(lines=(arc1, arc2),
                                             density=self.density,
                                             equinox_range=self.equinox_range,
                                             endpoint_dithering=self.endpoint_dithering,
                                             point_dithering=self.point_dithering)

        # transform
        return (Strokes.load_points(arc1, arc2, line)
                .rotate_points3d(degree_range=self.plane_rotate_degree_range)
                .move_points3d(dx_range=self.move_plane_range,
                               dz_range=self.move_plane_range,
                               dy_range=self.move_y_range))

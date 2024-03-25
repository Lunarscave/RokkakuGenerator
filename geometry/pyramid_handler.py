import random

from geometry import BaseGeometryHandler
from meta import Strokes
from utils import point_util


class PyramidHandler(BaseGeometryHandler):
    """
    Pyramid handler
    """

    def __init__(self):
        super().__init__()
        self.point_plane_range = None
        self.height_range = None
        self.bottom_y_range = None
        self.endpoint_dithering = None
        self.point_num_range = None

    def load_config(self) -> None:
        self.load_base_config()
        self.point_plane_range = self._config_["point_plane_range"]
        self.height_range = self._config_["height_range"]
        self.bottom_y_range = self._config_["bottom_y_range"]
        self.endpoint_dithering = self._config_["endpoint_dithering"]
        self.point_num_range = self._config_["point_num_range"]

    def name(self) -> str:
        return "pyramid"

    def generate_strokes(self) -> Strokes:
        # generate
        peak = point_util.get_point3d(x=0, z=0, y_range=self.height_range, dithering=self.endpoint_dithering)
        nums = random.randint(*self.point_num_range)
        bottom_points = [point_util.get_point3d(x_range=self.point_plane_range,
                                                z_range=self.point_plane_range,
                                                y_range=self.bottom_y_range,
                                                dithering=self.endpoint_dithering) for _ in range(0, nums)]
        line_axis_arr = (2, 1, 0)
        lines = [point_util.get_line3d(endpoints=(peak, bottom_points[i]),
                                       axis_arr=line_axis_arr,
                                       density=self.density,
                                       equinox_range=self.equinox_range,
                                       point_dithering=self.point_dithering) for i in range(0, nums)]

        # transform
        return (Strokes.load_points(*tuple(lines))
                .rotate_points3d(degree_range=self.plane_rotate_degree_range)
                .move_points3d(dx_range=self.move_plane_range,
                               dz_range=self.move_plane_range,
                               dy_range=self.move_y_range))

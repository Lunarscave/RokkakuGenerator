from typing import Tuple

from numpy import ndarray

from geometry import BaseGeometryHandler
from meta import Strokes, BuiltinGeometry
from utils import generate_point_util


class HipHandler(BaseGeometryHandler):
    """
    Hip handler
    """

    def __init__(self):
        """
        Init hip handler with rules.
        """
        super().__init__()
        self.point_bottom_plane_range: Tuple[float, float] | None = None
        self.point_top_x_range: Tuple[float, float] | None = None
        self.point_top_z_range: Tuple[float, float] | None = None
        self.height_range: Tuple[float, float] | None = None
        self.bottom_y_range: Tuple[float, float] | None = None
        self.endpoint_dithering: Tuple[float, float] | None = None

    def load_config(self) -> None:
        """
        Load hip handler config.
        """
        self.load_base_config()
        self.point_bottom_plane_range = self._config_["point_bottom_plane_range"]
        self.point_top_x_range = self._config_["point_top_x_range"]
        self.point_top_z_range = self._config_["point_top_z_range"]
        self.height_range = self._config_["height_range"]
        self.bottom_y_range = self._config_["bottom_y_range"]
        self.endpoint_dithering = self._config_["endpoint_dithering"]

    def prototype(self) -> BuiltinGeometry:
        """
        Get hip handler name.
        """
        return BuiltinGeometry.Hip

    def generate_strokes(self) -> Strokes:
        """
        Generate hip strokes.
        :return: Hip strokes.
        """

        # generate
        dx = [1, -1, -1, 1]
        dz = [1, 1, -1, -1]
        bottom_points = generate_point_util.get_point3d(
            x_range=self.point_bottom_plane_range,
            z_range=self.point_bottom_plane_range,
            y_range=self.bottom_y_range
        )
        top_point = generate_point_util.get_point3d(
            x_range=self.point_top_x_range,
            z_range=self.point_top_z_range,
            y_range=self.height_range
        )

        top_endpoints = [generate_point_util.get_point3d(
            x=float(dx[i] * top_point[0]),
            z=float(dz[i] * top_point[1]),
            y=float(top_point[2]),
            dithering=self.endpoint_dithering
        ) for i in range(0, 2)]
        bottom_endpoints = [generate_point_util.get_point3d(
            x=float(dx[i] * bottom_points[0]),
            z=float(dz[i] * bottom_points[1]),
            y=float(bottom_points[2]),
            dithering=self.endpoint_dithering
        ) for i in range(0, 4)]

        vertical_line_axis_arr = (2, 1, 0)
        top_index = [0, 1, 1, 0]
        vertical_lines = [generate_point_util.get_line3d(
            endpoints=(top_endpoints[top_index[i]], bottom_endpoints[i]),
            axis_arr=vertical_line_axis_arr,
            density=self.density,
            equinox_range=self.equinox_range,
            endpoint_dithering=self.endpoint_dithering,
            point_dithering=self.point_dithering
        ) for i in range(0, 4)]
        line = generate_point_util.get_line3d(
            endpoints=(top_endpoints[0], top_endpoints[1]),
            density=self.density,
            equinox_range=self.equinox_range,
            endpoint_dithering=self.endpoint_dithering,
            point_dithering=self.point_dithering
        )

        # transform
        return Strokes.load_points(
            *tuple(vertical_lines), line
        ).rotate_points3d(
            degree_range=self.plane_rotate_degree_range
        ).move_points3d(
            dx_range=self.move_plane_range,
            dz_range=self.move_plane_range,
            dy_range=self.move_y_range
        )

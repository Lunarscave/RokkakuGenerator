import random
from typing import Tuple

from numpy import ndarray

from geometry import BaseGeometryHandler
from meta import Strokes, BuiltinGeometry
from utils import generate_point_util


class CuboidHandler(BaseGeometryHandler):
    """
    Cuboid handler.
    """

    def __init__(self):
        """
        Init cuboid handler with rules.
        """
        super().__init__()
        self.point_plane_range: Tuple[float, float] | None = None
        self.height_range: Tuple[float, float] | None = None
        self.endpoint_equinox_range: Tuple[float, float] | None = None
        self.bottom_y_range: Tuple[float, float] | None = None
        self.dip_degree_range: Tuple[float, float] | None = None
        self.endpoint_dithering: Tuple[float, float] | None = None

    def load_config(self) -> None:
        """
        Load cuboid handler config.
        """
        self.load_base_config()
        self.point_plane_range = self._config_["point_plane_range"]
        self.height_range = self._config_["height_range"]
        self.endpoint_equinox_range = self._config_["endpoint_equinox_range"]
        self.bottom_y_range = self._config_["bottom_y_range"]
        self.dip_degree_range = self._config_["dip_degree_range"]
        self.endpoint_dithering = self._config_["endpoint_dithering"]

    def prototype(self) -> BuiltinGeometry:
        """
        Get cuboid handler name.
        """
        return BuiltinGeometry.Cuboid

    def generate_strokes(self) -> Strokes:
        """
        Generate cuboid strokes.
        :return: Cuboid strokes.
        """

        # generate
        dx = [1, -1, -1, 1]
        dz = [1, 1, -1, -1]
        top_line_axis = ((0, 1, 2), (1, 0, 2))
        point = generate_point_util.get_point3d(
            x_range=self.point_plane_range,
            z_range=self.point_plane_range,
            y_range=self.height_range
        )

        endpoints = [generate_point_util.get_point3d(
            x=float(dx[i] * point[0]),
            z=float(dz[i] * point[1]),
            y=float(point[2]),
            dithering=self.endpoint_dithering
        ) for i in range(0, 4)]

        top_line = [generate_point_util.get_line3d(
            endpoints=(endpoints[i], endpoints[(i + 1) % 4]),
            density=self.density,
            axis_arr=top_line_axis[i % 2],
            equinox_range=self.equinox_range,
            point_dithering=self.point_dithering
        ) for i in range(0, 4)]
        top_line = generate_point_util.concatenate_points(*tuple(top_line))

        vertical_nums = random.randint(0, 4)
        if vertical_nums == 0:
            index = random.randint(0, 3)
            vertical_lines = [generate_point_util.get_vertical_line3d_by_endpoint(
                endpoint=endpoints[index],
                bottom_y_range=self.bottom_y_range,
                density=self.density,
                equinox_range=self.equinox_range,
                endpoint_dithering=self.endpoint_dithering,
                point_dithering=self.point_dithering
            )]
        else:
            vertical_lines = [generate_point_util.get_vertical_line3d_by_line(
                endpoints=(endpoints[i], endpoints[(i + 1) % 4]),
                bottom_y_range=self.bottom_y_range,
                degree_range=self.dip_degree_range,
                density=self.density,
                endpoint_equinox_range=self.endpoint_equinox_range,
                equinox_range=self.equinox_range,
                endpoint_dithering=self.endpoint_dithering,
                point_dithering=self.point_dithering
            ) for i in range(0, vertical_nums)]

        # transform
        return Strokes.load_points(
            top_line,
            *tuple(vertical_lines)
        ).rotate_points3d(
            degree_range=self.plane_rotate_degree_range
        ).move_points3d(
            dx_range=self.move_plane_range,
            dz_range=self.move_plane_range,
            dy_range=self.move_y_range
        )

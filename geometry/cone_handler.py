from typing import Tuple

from geometry import BaseGeometryHandler
from meta import Strokes, BuiltinGeometry
from utils import generate_point_util


class ConeHandler(BaseGeometryHandler):
    """
    Cone handler.
    """

    def __init__(self):
        """
        Init cone handler with rules.
        """
        super().__init__()
        self.radius_range: Tuple[float, float] | None = None
        self.height_range: Tuple[float, float] | None = None
        self.endpoint_dithering: Tuple[float, float] | None = None

    def load_config(self) -> None:
        """
        Load cone handler config.
        """
        self.load_base_config()
        self.radius_range = self._config_["radius_range"]
        self.height_range = self._config_['height_range']
        self.endpoint_dithering = self._config_['endpoint_dithering']

    def prototype(self) -> BuiltinGeometry:
        """
        Get cone handler name.
        """
        return BuiltinGeometry.Cone

    def generate_strokes(self) -> Strokes:
        """
        Generate cone strokes.
        :return: Cone strokes.
        """

        # generate
        point1 = generate_point_util.get_point3d(x_range=self.radius_range, z=0, y=0)
        point2 = -point1
        peak = generate_point_util.get_point3d(x=0, z=0, y_range=self.height_range, dithering=self.endpoint_dithering)
        line_axis_arr = (2, 1, 0)
        line1 = generate_point_util.get_line3d(
            endpoints=(point1, peak),
            density=self.density,
            axis_arr=line_axis_arr,
            equinox_range=self.equinox_range,
            point_dithering=self.point_dithering
        )
        line2 = generate_point_util.get_line3d(
            endpoints=(point2, peak),
            density=self.density,
            axis_arr=line_axis_arr,
            equinox_range=self.equinox_range,
            point_dithering=self.point_dithering
        )

        # transform
        return Strokes.load_points(
            line1, line2
        ).rotate_points3d(
            degree_range=self.plane_rotate_degree_range
        ).move_points3d(
            dx_range=self.move_plane_range,
            dz_range=self.move_plane_range,
            dy_range=self.move_y_range
        )

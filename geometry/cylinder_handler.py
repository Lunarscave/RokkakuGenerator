from typing import Tuple

from geometry import BaseGeometryHandler
from meta import Strokes, BuiltinGeometry
from utils import generate_point_util


class CylinderHandler(BaseGeometryHandler):
    """
    Cylinder handler
    """

    def __init__(self):
        """
        Init cylinder handler with rules.
        """
        super().__init__()
        self.a_range: Tuple[float, float] | None = None
        self.b_range: Tuple[float, float] | None = None
        self.height_range: Tuple[float, float] | None = None
        self.bottom_y_range: Tuple[float, float] | None = None
        self.dip_degree_range: Tuple[float, float] | None = None
        self.endpoint_dithering: Tuple[float, float] | None = None

    def load_config(self) -> None:
        """
        Load cylinder handler config.
        """
        self.load_base_config()
        self.a_range = self._config_["a_range"]
        self.b_range = self._config_["b_range"]
        self.height_range = self._config_["height_range"]
        self.bottom_y_range = self._config_["bottom_y_range"]
        self.dip_degree_range = self._config_["dip_degree_range"]
        self.endpoint_dithering = self._config_["endpoint_dithering"]

    def prototype(self) -> BuiltinGeometry:
        """
        Get cylinder handler name.
        """
        return BuiltinGeometry.Cylinder

    def generate_strokes(self) -> Strokes:
        """
        Generate cylinder strokes.
        :return: Cylinder strokes.
        """

        # generate
        ellipse = generate_point_util.get_ellipse_curve3d(
            a_range=self.a_range,
            b_range=self.b_range,
            y_range=self.height_range,
            density=self.density,
            point_dithering=self.point_dithering
        )
        line = generate_point_util.get_vertical_line3d_by_curve(
            points=ellipse,
            bottom_y_range=self.bottom_y_range,
            degree_range=self.dip_degree_range,
            density=self.density,
            equinox_range=self.equinox_range,
            endpoint_dithering=self.endpoint_dithering,
            point_dithering=self.point_dithering
        )

        # transform
        return Strokes.load_points(
            ellipse, line
        ).rotate_points3d(
            degree_range=self.plane_rotate_degree_range
        ).move_points3d(
            dx_range=self.move_plane_range,
            dz_range=self.move_plane_range,
            dy_range=self.move_y_range
        )

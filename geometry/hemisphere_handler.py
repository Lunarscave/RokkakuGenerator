from typing import Tuple

from numpy import ndarray

from geometry import BaseGeometryHandler
from meta import Strokes, BuiltinGeometry
from utils import generate_point_util


class HemisphereHandler(BaseGeometryHandler):
    """
    Hemisphere handler
    """

    def __init__(self):
        """
        Init hemisphere handler with rules.
        """
        super().__init__()
        self.r_range: Tuple[float, float] | None = None
        self.th_degree_range: Tuple[float, float] | None = None
        self.bottom_y_range: Tuple[float, float] | None = None

    def load_config(self) -> None:
        """
        Load hemisphere handler config.
        """
        self.load_base_config()
        self.r_range = self._config_["r_range"]
        self.th_degree_range = self._config_["th_degree_range"]
        self.bottom_y_range = self._config_["bottom_y_range"]

    def prototype(self) -> BuiltinGeometry:
        """
        Get hemisphere handler name.
        """
        return BuiltinGeometry.Hemisphere

    def generate_strokes(self) -> Strokes:
        """
        Generate hemisphere strokes.
        :return: Hemisphere strokes.
        """

        # generate
        arc = generate_point_util.get_vertical_arc(
            density=self.density,
            r_range=self.r_range,
            th_degree_range=self.th_degree_range,
            y_range=self.bottom_y_range,
            point_dithering=self.point_dithering
        )

        # transform
        return Strokes.load_points(
            arc
        ).rotate_points3d(
            degree_range=self.plane_rotate_degree_range
        ).move_points3d(
            dx_range=self.move_plane_range,
            dz_range=self.move_plane_range,
            dy_range=self.move_y_range
        )

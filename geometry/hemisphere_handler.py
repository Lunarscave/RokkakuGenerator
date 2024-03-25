from numpy import ndarray

from geometry import BaseGeometryHandler
from meta import Strokes
from utils import point_util


class HemisphereHandler(BaseGeometryHandler):
    """
    Hemisphere handler
    """

    def __init__(self):
        super().__init__()
        self.r_range = None
        self.th_degree_range = None
        self.bottom_y_range = None

    def load_config(self) -> None:
        self.load_base_config()
        self.r_range = self._config_["r_range"]
        self.th_degree_range = self._config_["th_degree_range"]
        self.bottom_y_range = self._config_["bottom_y_range"]

    def name(self) -> str:
        return "hemisphere"

    def generate_strokes(self) -> ndarray:
        # generate
        arc = point_util.get_vertical_arc(density=self.density,
                                          r_range=self.r_range,
                                          th_degree_range=self.th_degree_range,
                                          y_range=self.bottom_y_range,
                                          point_dithering=self.point_dithering)

        # transform
        return (Strokes.load_points(arc)
                .rotate_points3d(degree_range=self.plane_rotate_degree_range)
                .move_points3d(dx_range=self.move_plane_range,
                               dz_range=self.move_plane_range,
                               dy_range=self.move_y_range))

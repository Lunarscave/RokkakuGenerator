from numpy import ndarray

from geometry import BaseGeometryHandler
from meta import Strokes
from utils import point_util


class CylinderHandler(BaseGeometryHandler):
    """
    Cylinder handler
    """

    def __init__(self):
        super().__init__()
        self.a_range = None
        self.b_range = None
        self.height_range = None
        self.bottom_y_range = None
        self.dip_degree_range = None
        self.endpoint_dithering = None

    def load_config(self) -> None:
        self.load_base_config()
        self.a_range = self._config_["a_range"]
        self.b_range = self._config_["b_range"]
        self.height_range = self._config_["height_range"]
        self.bottom_y_range = self._config_["bottom_y_range"]
        self.dip_degree_range = self._config_["dip_degree_range"]
        self.endpoint_dithering = self._config_["endpoint_dithering"]

    def name(self) -> str:
        return "cylinder"

    def generate_strokes(self) -> ndarray:
        # generate
        ellipse = point_util.get_ellipse_curve3d(a_range=self.a_range,
                                                 b_range=self.b_range,
                                                 y_range=self.height_range,
                                                 density=self.density,
                                                 point_dithering=self.point_dithering)
        line = point_util.get_vertical_line3d_by_curve(points=ellipse,
                                                       bottom_y_range=self.bottom_y_range,
                                                       degree_range=self.dip_degree_range,
                                                       density=self.density,
                                                       equinox_range=self.equinox_range,
                                                       endpoint_dithering=self.endpoint_dithering,
                                                       point_dithering=self.point_dithering)

        # transform
        return (Strokes.load_points(ellipse, line)
                .rotate_points3d(degree_range=self.plane_rotate_degree_range)
                .move_points3d(dx_range=self.move_plane_range,
                               dz_range=self.move_plane_range,
                               dy_range=self.move_y_range))

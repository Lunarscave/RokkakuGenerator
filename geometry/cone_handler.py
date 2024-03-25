from geometry import BaseGeometryHandler
from meta import Strokes
from utils import point_util


class ConeHandler(BaseGeometryHandler):
    """
    Cone handler
    """

    def __init__(self):
        super().__init__()
        self.radius_range = None
        self.height_range = None
        self.endpoint_dithering = None

    def load_config(self) -> None:
        self.load_base_config()
        self.radius_range = self._config_["radius_range"]
        self.height_range = self._config_['height_range']
        self.endpoint_dithering = self._config_['endpoint_dithering']

    def name(self) -> str:
        return "cone"

    def generate_strokes(self) -> Strokes:
        # generate
        point1 = point_util.get_point3d(x_range=self.radius_range, z=0, y=0)
        point2 = -point1
        peak = point_util.get_point3d(x=0, z=0, y_range=self.height_range, dithering=self.endpoint_dithering)
        line_axis_arr = (2, 1, 0)
        line1 = point_util.get_line3d(endpoints=(point1, peak),
                                      density=self.density,
                                      axis_arr=line_axis_arr,
                                      equinox_range=self.equinox_range,
                                      point_dithering=self.point_dithering)
        line2 = point_util.get_line3d(endpoints=(point2, peak),
                                      density=self.density,
                                      axis_arr=line_axis_arr,
                                      equinox_range=self.equinox_range,
                                      point_dithering=self.point_dithering)

        # transform
        return (Strokes.load_points(line1, line2)
                .rotate_points3d(degree_range=self.plane_rotate_degree_range)
                .move_points3d(dx_range=self.move_plane_range,
                               dz_range=self.move_plane_range,
                               dy_range=self.move_y_range))

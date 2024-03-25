import math
import random

import numpy as np

from geometry import BaseGeometryHandler
from meta import FreeformStrokes
from utils import point_util


class FreeformHandler(BaseGeometryHandler):
    """
    Freeform handler
    """

    def __init__(self):
        super().__init__()
        self.point_num_range = None
        self.point_plane_range = None
        self.height_range = None
        self.bottom_y_range = None
        self.curve_poss = None
        self.dip_degree_range = None
        self.line_endpoint_dithering = None
        self.curve_endpoint_dithering = None

    def load_config(self) -> None:
        self.load_base_config()
        self.point_num_range = self._config_["point_num_range"]
        self.point_plane_range = self._config_["point_plane_range"]
        self.height_range = self._config_["height_range"]
        self.bottom_y_range = self._config_["bottom_y_range"]
        self.curve_poss = self._config_["curve_poss"]
        self.dip_degree_range = self._config_["dip_degree_range"]
        self.line_endpoint_dithering = self._config_["line_endpoint_dithering"]
        self.curve_endpoint_dithering = self._config_["curve_endpoint_dithering"]

    def name(self) -> str:
        return "freeform"

    def generate_strokes(self) -> FreeformStrokes:
        # generate
        endpoint_nums = random.randint(*self.point_num_range)
        height = random.uniform(*self.height_range)
        top_endpoints = np.array([point_util.get_point3d(x_range=self.point_plane_range,
                                                         z_range=self.point_plane_range,
                                                         y=height)
                                  for _ in range(0, endpoint_nums)])
        mean_top_points = np.array([np.mean(top_endpoints[:, 0]), np.mean(top_endpoints[:, 1]), 0])
        top_endpoints -= mean_top_points
        top_endpoints = list(top_endpoints)
        top_endpoints.sort(key=lambda p: math.atan2(p[1], p[0]))
        top_endpoints = np.array(top_endpoints)

        top_points = []
        top_points_types = []
        endpoint_ditherings = [self.curve_endpoint_dithering, self.line_endpoint_dithering]
        for i in range(0, endpoint_nums):
            line_type = 0 if random.uniform(0, 1) <= self.curve_poss else 1
            points = point_util.get_curve3d(endpoints=(top_endpoints[i], top_endpoints[(i + 1) % endpoint_nums]),
                                            density=self.density,
                                            equinox_range=self.equinox_range,
                                            endpoint_dithering=endpoint_ditherings[line_type],
                                            point_dithering=self.point_dithering)
            top_points.extend(list(points))
            top_points_types.extend([(line_type + 1) if j != 0 else 0 for j, _ in enumerate(points)])

        top_points = np.array(top_points)

        top_endpoints_index = []
        for i, v in enumerate(top_points_types):
            if v == 0:
                top_endpoints_index.append(i)

        vertical_nums = random.randint(0, endpoint_nums)
        if vertical_nums == 0:
            vert_points = [point_util.get_vertical_line3d_by_endpoint(endpoint=top_endpoints[top_endpoints_index[0]],
                                                                      bottom_y_range=self.bottom_y_range,
                                                                      density=self.density,
                                                                      equinox_range=self.equinox_range,
                                                                      endpoint_dithering=self.line_endpoint_dithering,
                                                                      point_dithering=self.point_dithering)]
        else:
            print()
            vert_points = [
                point_util.get_vertical_line3d_by_curve(points=top_points[top_endpoints_index[i]:
                                                                          top_endpoints_index[i + 1]
                                                                          if i + 1 < endpoint_nums else -1],
                                                        bottom_y_range=self.bottom_y_range,
                                                        degree_range=self.dip_degree_range,
                                                        density=self.density,
                                                        equinox_range=self.equinox_range,
                                                        endpoint_dithering=self.line_endpoint_dithering,
                                                        point_dithering=self.point_dithering)
                for i in range(0, vertical_nums)]

        # transform
        return (FreeformStrokes.load_points(top_points, *tuple(vert_points))
                .load_plane_strokes_types(top_points_types)
                .rotate_points3d(degree_range=self.plane_rotate_degree_range)
                .move_points3d(dx_range=self.move_plane_range,
                               dz_range=self.move_plane_range,
                               dy_range=self.move_y_range))

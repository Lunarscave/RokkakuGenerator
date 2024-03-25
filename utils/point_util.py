import math
import random
from typing import Any

import numpy as np
from numpy import ndarray, linalg


def _get_value_(value: float,
                value_range: tuple) -> float:
    return random.uniform(*value_range) if value is None else value


def get_point_distance_3d(endpoints: tuple) -> ndarray:
    """
    get point distance in 3d
    """
    return np.linalg.norm(endpoints[0] - endpoints[1])


def spherical_2cartesian3d(r: float,
                           th: float,
                           phi: float,
                           base: ndarray = np.array([0, 0, 0])) -> ndarray:
    """
    transform spherical coordinate to cartesian cartesian in 3d
    """
    return np.array([r * math.cos(phi) * math.cos(th), r * math.cos(phi) * math.sin(th), r * math.sin(phi)]) + base


def get_point3d(x: float = None,
                y: float = None,
                z: float = None,
                x_range: tuple = (0, 0),
                z_range: tuple = (0, 0),
                y_range: tuple = (0, 0),
                dithering: tuple = (0, 0)) -> ndarray:
    """
    get point in 3d
    """
    values = [x, z, y]
    ranges = [x_range, z_range, y_range]
    point = np.array([_get_value_(values[i], ranges[i]) for i, _ in enumerate(ranges)])
    return dither_point3d(point, dithering)


def dither_point3d(point: ndarray,
                   dithering: tuple = (0, 0)) -> ndarray:
    """
    dither point in 3d
    """
    r = random.uniform(*dithering)
    th = random.uniform(0, 2 * math.pi)
    phi = random.uniform(0, 2 * math.pi)
    return point + spherical_2cartesian3d(r, th, phi)


def dither_points3d(points: ndarray,
                    dithering: tuple = (0, 0)) -> ndarray:
    """
    dither point in 3d
    """
    return np.array([dither_point3d(point, dithering) for point in points])


def get_equinox3d(endpoints: tuple,
                  equinox: float = None,
                  equinox_range: tuple = (0, 1),
                  dithering: tuple = (0, 0)) -> ndarray:
    """
    get equinox point of line, and the division start from line[0]
    """
    equinox = _get_value_(equinox, equinox_range)
    point = np.array([endpoints[0][i] * (1 - equinox) + endpoints[1][i] * equinox for i in range(0, 3)])
    return dither_point3d(point, dithering)


def get_bezier_curve3d(control_points: ndarray,
                       t_arr: ndarray,
                       dithering: tuple = (0, 0),
                       uniform_velocity: bool = False) -> ndarray:
    """
    Get Bézier curve with control points through bernstein poly. And can choose the uniform velocity features.
    """

    def bernstein(i: int,
                  n: int,
                  t: ndarray) -> float:
        return math.comb(n, i) * (t ** (n - i)) * (1 - t) ** i

    def get_bezier_speed(c_points: ndarray,
                         t: float) -> float:
        n = len(c_points) - 1
        velocities = np.array([[math.comb(n, j) * p[i] *
                                ((j * t ** (j - 1) if t != 0 else 0)
                                 + ((n - j) * (1 - t) ** (n - j - 1) if 1 - t != 0 else 0))
                                for j, p in enumerate(c_points)] for i in range(0, 3)])
        return np.linalg.norm(velocities)

    def get_bezier_length(c_points: ndarray,
                          t: float,
                          index: int) -> float:
        if index % 2 == 0:
            index += 1
        dt = t / index
        half_index = index // 2
        sum1 = np.sum([get_bezier_speed(c_points, (2 * i + 1) * dt) for i in range(0, half_index)])
        sum2 = np.sum([get_bezier_speed(c_points, 2 * i * dt) for i in range(0, half_index)])
        return (get_bezier_speed(c_points, 0) + get_bezier_speed(c_points, 1) + 4 * sum1 + 2 * sum2) * dt / 3

    def bezier_t2real_time(c_points: ndarray,
                           arc_length: float,
                           index: int) -> float:
        real_time = 0
        low_time = 0
        high_time = 1
        d_time = 0
        d_length = 0
        epsilon = 0.001
        start_flag = True
        while start_flag or (math.fabs(d_length) > epsilon and math.fabs(d_time) > epsilon):
            start_flag = False
            d_time = - (real_time - low_time) / 2 if d_length > 0 else (high_time - real_time) / 2
            real_time += d_time
            real_time_length = get_bezier_length(c_points, real_time, index)
            d_length = real_time_length - arc_length
            if d_length > 0:
                high_time = real_time
            else:
                low_time = real_time
        return real_time

    control_nums = len(control_points)
    if uniform_velocity:
        t_nums = len(t_arr)
        length = get_bezier_length(control_points, 1, t_nums - 1)
        t_arr = np.array([bezier_t2real_time(control_points, i / t_nums * length, i) for i in range(0, t_nums)])
    bernstein_arr = np.array([bernstein(i, control_nums - 1, t_arr) for i in range(0, control_nums)])
    points = np.matmul(control_points.T, bernstein_arr).T
    return dither_points3d(points, dithering)


def get_ellipse_curve3d(a: float = None,
                        b: float = None,
                        y: float = None,
                        a_range: tuple = (1, 1),
                        b_range: tuple = (1, 1),
                        y_range: tuple = (0, 0),
                        density: float = 1.0,
                        point_dithering: tuple = (0, 0)) -> ndarray:
    """
    Get standard elliptic curve
    """
    a = _get_value_(a, a_range)
    b = _get_value_(b, b_range)
    y = _get_value_(y, y_range)
    nums = math.floor(2 * math.pi * b + 4 * (a - b) / density)
    points = [[a * math.cos(i / nums * 2 * math.pi), b * math.sin(i / nums * 2 * math.pi), y] for i in range(0, nums)]
    return dither_point3d(np.array(points), point_dithering)


def get_curve3d(endpoints: tuple,
                axis_arr: tuple = (0, 1, 2),
                density: float = 1.0,
                equinox_range: tuple = (0, 1),
                endpoint_dithering: tuple = (0, 0),
                point_dithering: tuple = (0, 0),
                uniform_velocity: bool = False) -> ndarray:
    """
    get curve through bernstein poly
    """
    control_points = np.array([get_equinox3d(endpoints=endpoints,
                                             equinox_range=equinox_range,
                                             dithering=endpoint_dithering) for _ in range(0, 3)])
    control_points = sort_points3d(points=control_points,
                                   axis_arr=axis_arr,
                                   reverse_arr=tuple((endpoints[0] - endpoints[1]) > 0))
    control_points = np.insert(control_points, 0, endpoints[0], axis=0)
    control_points = np.insert(control_points, len(control_points), endpoints[1], axis=0)
    t_arr = np.linspace(0.0, 1.0, int(get_point_distance_3d(endpoints) / density))
    return get_bezier_curve3d(control_points, t_arr, point_dithering, uniform_velocity)


def get_line3d(endpoints: tuple,
               axis_arr: tuple = (0, 1, 2),
               density: float = 1.0,
               equinox_range: tuple = (0, 1),
               endpoint_dithering: tuple = (0, 0),
               point_dithering: tuple = (0, 0),
               uniform_velocity: bool = False) -> ndarray:
    """
    get line through bernstein poly
    """
    return get_curve3d(endpoints, axis_arr, density, equinox_range, endpoint_dithering, point_dithering,
                       uniform_velocity)


def get_connect_line3d(lines: tuple,
                       axis_arr: tuple = (0, 1, 2),
                       density: float = 1.0,
                       equinox_range: tuple = (0, 1),
                       endpoint_dithering: tuple = (0, 0),
                       point_dithering: tuple = (0, 0),
                       uniform_velocity: bool = False) -> ndarray:
    index1 = random.randint(0, len(lines[0]) - 1)
    index2 = random.randint(0, len(lines[1]) - 1)
    return get_line3d(endpoints=(lines[0][index1], lines[1][index2]),
                      density=density,
                      axis_arr=axis_arr,
                      equinox_range=equinox_range,
                      endpoint_dithering=endpoint_dithering,
                      point_dithering=point_dithering,
                      uniform_velocity=uniform_velocity)


def _get_vertical_line3e_by_norm_vector_(endpoints: tuple,
                                         normal_vector: ndarray = np.array([0, 0, 0]),
                                         radian_range: tuple = None,
                                         degree_range: tuple = (0, 0),
                                         density: float = 1.0,
                                         equinox_range: tuple = (0, 1),
                                         point_dithering: tuple = (0, 0),
                                         uniform_velocity: bool = False) -> ndarray:
    """
    Get vertical line by endpoint and normal vector through bernstein poly
    """
    radian_range = tuple(np.deg2rad(degree_range)) if radian_range is None else radian_range
    dip = random.uniform(*radian_range)
    bottom_point = endpoints[1] + np.multiply(normal_vector, math.tan(dip) * (endpoints[0][2] - endpoints[1][2]))

    return get_line3d(endpoints=(endpoints[0], bottom_point),
                      density=density,
                      axis_arr=(2, 1, 0),
                      equinox_range=equinox_range,
                      point_dithering=point_dithering,
                      uniform_velocity=uniform_velocity)


def get_vertical_line3d_by_endpoint(endpoint: ndarray,
                                    bottom_y_range: tuple,
                                    bottom_y: float = None,
                                    density: float = 1.0,
                                    equinox_range: tuple = (0, 1),
                                    endpoint_dithering: tuple = (0, 0),
                                    point_dithering: tuple = (0, 0),
                                    uniform_velocity: bool = False) -> ndarray:
    """
    Get vertical line by endpoint through bernstein poly
    """
    point1 = get_point3d(x=float(endpoint[0]),
                         z=float(endpoint[1]),
                         y=float(endpoint[2]),
                         dithering=endpoint_dithering)
    point2 = get_point3d(x=float(point1[0]),
                         z=float(point1[1]),
                         y=bottom_y,
                         y_range=bottom_y_range,
                         dithering=endpoint_dithering)

    return _get_vertical_line3e_by_norm_vector_(endpoints=(point1, point2),
                                                density=density,
                                                equinox_range=equinox_range,
                                                point_dithering=point_dithering,
                                                uniform_velocity=uniform_velocity)


def get_vertical_line3d_by_line(endpoints: tuple,
                                bottom_y_range: tuple,
                                bottom_y: float = None,
                                radian_range: tuple = None,
                                degree_range: tuple = (0, 0),
                                density: float = 1.0,
                                endpoint_equinox_range: tuple = (0, 1),
                                equinox_range: tuple = (0, 1),
                                endpoint_dithering: tuple = (0, 0),
                                point_dithering: tuple = (0, 0),
                                uniform_velocity: bool = False) -> ndarray:
    """
    Get vertical line by line through bernstein poly
    """
    point1 = get_equinox3d(endpoints=endpoints,
                           equinox_range=endpoint_equinox_range,
                           dithering=endpoint_dithering)
    point2 = get_point3d(x=float(point1[0]),
                         z=float(point1[1]),
                         y=bottom_y,
                         y_range=bottom_y_range,
                         dithering=endpoint_dithering)

    th = math.atan2(endpoints[1][1] - endpoints[0][1], endpoints[1][0] - endpoints[0][0])
    th += 90 if th < 0 else -90
    v = np.array([math.sin(th), math.cos(th), 0])

    return _get_vertical_line3e_by_norm_vector_(endpoints=(point1, point2),
                                                normal_vector=v,
                                                radian_range=radian_range,
                                                degree_range=degree_range,
                                                density=density,
                                                equinox_range=equinox_range,
                                                point_dithering=point_dithering,
                                                uniform_velocity=uniform_velocity)


def get_vertical_line3d_by_curve(points: ndarray,
                                 bottom_y_range: tuple,
                                 bottom_y: float = None,
                                 radian_range: tuple = None,
                                 degree_range: tuple = (0, 0),
                                 density: float = 1.0,
                                 equinox_range: tuple = (0, 1),
                                 endpoint_dithering: tuple = (0, 0),
                                 point_dithering: tuple = (0, 0),
                                 uniform_velocity: bool = False) -> ndarray:
    """
    Get vertical line by line through bernstein poly
    """

    def get_curve_normal_vector(endpoints: ndarray) -> ndarray:
        """
        Get curve normal vector by fitting parabola
        """
        px = endpoints[:, 0]
        pz = endpoints[:, 1]
        ta = linalg.norm([px[1] - px[0], pz[1] - pz[0]])
        tb = linalg.norm([px[2] - px[1], pz[2] - pz[1]])

        mat = np.array([[1, -ta, ta ** 2], [1, 0, 0], [1, tb, tb ** 2]])

        a = np.matmul(linalg.pinv(mat), px)
        b = np.matmul(linalg.pinv(mat), pz)

        return np.insert(np.array([b[1], -a[1]]) / np.sqrt(a[1] ** 2. + b[1] ** 2.), 2, 0)

    point_length = len(points)
    i = random.randint(0, point_length - 1)
    point1 = dither_point3d(points[i], endpoint_dithering)
    point2 = get_point3d(x=float(point1[0]),
                         z=float(point1[1]),
                         y=bottom_y,
                         y_range=bottom_y_range,
                         dithering=endpoint_dithering)

    fit_points = np.array([points[(i - 1 + point_length) % point_length], points[i], points[(i + 1) % point_length]])
    v = get_curve_normal_vector(fit_points)

    return _get_vertical_line3e_by_norm_vector_(endpoints=(point1, point2),
                                                normal_vector=v,
                                                radian_range=radian_range,
                                                degree_range=degree_range,
                                                density=density,
                                                equinox_range=equinox_range,
                                                point_dithering=point_dithering,
                                                uniform_velocity=uniform_velocity)


def get_vertical_arc(density: float = 1.0,
                     r_range: tuple = (1, 1),
                     th_radian_range: tuple = None,
                     th_degree_range: tuple = (90, 90),
                     rot_radian_range: tuple = None,
                     rot_degree_range: tuple = (0, 0),
                     y_range: tuple = (0, 0),
                     base: ndarray = np.array([0, 0, 0]),
                     r: float = None,
                     th: float = None,
                     rot: float = None,
                     y: float = None,
                     point_dithering: tuple = (0, 0)) -> ndarray:
    """
    Get vertical arc
    """
    th_radian_range = tuple(np.deg2rad(th_degree_range)) if th_radian_range is None else th_radian_range
    rot_radian_range = tuple(np.deg2rad(rot_degree_range)) if rot_radian_range is None else rot_radian_range

    r = _get_value_(r, r_range)
    th = _get_value_(th, th_radian_range)
    rot = _get_value_(rot, rot_radian_range)
    y = _get_value_(y, y_range)

    nums = math.floor(r * th / density)
    center = np.array([0, 0, -r * math.sin((math.pi - th) / 2)])
    points = np.array([np.array([r * math.cos(i / nums * th + (math.pi - th) / 2) * math.sin(rot),
                                 r * math.cos(i / nums * th + (math.pi - th) / 2) * math.cos(rot),
                                 r * math.sin(i / nums * th + (math.pi - th) / 2) + y]) + center + base
                       for i in range(0, nums)])
    return dither_points3d(points, point_dithering)


def sort_points3d(points: ndarray,
                  axis_arr: tuple = (0, 1, 2),
                  reverse_arr: tuple[bool] = np.array([False, False, False])) -> ndarray:
    """
    Sort points along axis(and choose reverse)
    """
    point_list = list(points)
    require_length = len(axis_arr)
    point_list.sort(key=lambda p: tuple([-p[axis_arr[i]] if reverse_arr[i] else p[axis_arr[i]]
                                        for i in range(0, require_length)]))
    return np.array(point_list)


def concatenate_points(*points_arrs: ndarray) -> ndarray:
    """
    Concatenate all points
    """
    return np.concatenate(list(points_arrs))

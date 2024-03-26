import logging
import os.path
import re

import numpy as np
import toml
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import ndarray

from geometry import BaseGeometryHandler, ConeHandler, CuboidHandler, CylinderHandler, HemisphereHandler, \
    PyramidHandler, ShedHandler, PlatformHandler, HipHandler, FreeformHandler
from meta import Strokes, FreeformStrokes
from utils import assert_util, hdf5_util


class StrokesGenerator:
    """
    Generate strokes
    """

    def __init__(self,
                 config_file_path: str = "config.toml") -> None:
        """
        Init the strokes generator with configuration
        """
        self.__geometry_map__ = {}
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(filename)s - %(message)s")
        self.__LOGGER__ = logging.getLogger()
        self.__generator_config__ = {}

        assert_util.is_not_none(re.match(r".*\.toml", config_file_path),
                                "extension of config file name '{0}' is not 'toml'.", config_file_path)
        self.__generator_config__ = toml.load(config_file_path)
        self.__load_geometries__()

    def load_geometry(self,
                      geometry: BaseGeometryHandler) -> None:
        """
        Load geometry with configuration
        """
        geometry_name = geometry.name()
        assert_util.is_true(geometry_name is not None and len(geometry_name) > 0, "geometry name should not be blank.")
        assert_util.is_false(geometry_name in self.__geometry_map__, "geometry {0} duplicated.", geometry_name)
        assert_util.is_true(geometry_name in self.__generator_config__.keys(),
                            "can not find geometry config of '{0}'", geometry_name)
        geometry.set_config(self.__generator_config__[geometry_name])

        geometry.validate()
        self.__geometry_map__[geometry_name] = geometry
        self.__LOGGER__.info(f"loaded geometry: {geometry_name}.")

    def __load_geometries__(self) -> None:
        """
        Load geometries with configuration
        """
        self.load_geometry(ConeHandler())
        self.load_geometry(CuboidHandler())
        self.load_geometry(CylinderHandler())
        self.load_geometry(HemisphereHandler())
        self.load_geometry(PyramidHandler())
        self.load_geometry(ShedHandler())
        self.load_geometry(PlatformHandler())
        self.load_geometry(HipHandler())
        self.load_geometry(FreeformHandler())

    def get_geometry_strokes(self,
                             geometry_name: str,
                             output_path: str = None) -> Strokes:
        """
        Get geometry strokes and save strokes
        """
        geometry = self.get_geometry(geometry_name)
        geometry.validate()
        geometry.load_config()
        strokes = geometry.generate_strokes()

        if output_path is not None:
            datas = [strokes.get_value()]
            data_names = ["value"]
            dtypes = ["float"]
            if isinstance(strokes, FreeformStrokes):
                datas.append(strokes.get_types())
                data_names.append("types")
                dtypes.append("int")
            hdf5_util.save_file(output_path, datas, data_names, dtypes)

        self.__LOGGER__.info(f"generated geometry: {geometry_name}.")

        return strokes

    def get_all_geometries_strokes(self,
                                   output_path: str = None,
                                   nums: int = 1) -> dict[str, list[Strokes]]:
        """
        Get all geometries strokes and save all strokes
        """
        names = self.list_geometries_name()
        strokes_map = {}
        for name in names:
            strokes_map[name] = []

        for i in range(0, nums):
            self.__LOGGER__.info(f"start to generate all strokes (turns: {i + 1})")
            for name in names:
                strokes_arr = strokes_map[name]
                strokes_arr.append(self.get_geometry_strokes(name))
                strokes_map[name] = strokes_arr

        if output_path is not None:
            hdf5_util.validate_directory_path(output_path)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            for name in names:
                datas = [[strokes.get_value(), strokes.get_types()] if isinstance(strokes, FreeformStrokes)
                         else [strokes.get_value()] for strokes in strokes_map[name]]
                data_names = [["value", "types"] if isinstance(strokes, FreeformStrokes)
                              else ["value"] for strokes in strokes_map[name]]
                dtypes = [["float", "int"] if isinstance(strokes, FreeformStrokes)
                          else ["float"] for strokes in strokes_map[name]]
                hdf5_util.save_files(directory_path=f"{output_path}\\{name}",
                                     datas=datas, data_names=data_names, dtypes=dtypes)

                self.__LOGGER__.info(f"generated geometry saved: {name}.")

        return strokes_map

    def get_geometry(self,
                     geometry_name: str) -> BaseGeometryHandler:
        """
        Get geometry from the geometry map
        """
        geometry = self.__geometry_map__.get(geometry_name)
        assert_util.is_not_none(geometry, f"geometry '{geometry_name}' is not found.")
        return geometry

    def list_geometries_name(self) -> list[str]:
        """
        List all geometries name
        """
        return list(self.__geometry_map__.keys())


if __name__ == "__main__":
    # main demo
    strokes_generator = StrokesGenerator()
    points = strokes_generator.get_geometry_strokes("freeform").get_value()
    # print(points)
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_xlabel('X label')
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')

    # ax.set_xlim(0, 100)
    # ax.set_ylim(0, 100)
    ax.set_zlim(0, 50)

    fig.add_axes(ax)
    plt.show()


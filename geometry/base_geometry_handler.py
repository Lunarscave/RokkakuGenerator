import os.path
from abc import abstractmethod, ABCMeta
from typing import Any

import toml
from numpy import ndarray

from meta import Strokes
from utils import assert_util


class BaseGeometryHandler(metaclass=ABCMeta):
    """
    Base geometry for generation
    """

    def __init__(self):
        """
        Init the geometry with rules
        """
        super().__init__()
        self.__RULES_PATH__ = "./geometry/rules.toml"
        self.__rules__ = {}
        self._config_ = {}

        self.density = None
        self.equinox_range = None
        self.point_dithering = None
        self.move_plane_range = None
        self.move_y_range = None
        self.plane_rotate_degree_range = None

        self.__load_rules__()

    @abstractmethod
    def name(self) -> str:
        """
        Get the geometry name
        """
        pass

    def validate(self) -> None:
        """
        Validate the config and the name of geometry
        """
        name = self.name()
        assert_util.is_true(name is not None and len(name) > 0, "geometry name can not be blank.")
        assert_util.is_true(isinstance(self._config_, dict) and self._config_ != {},
                            "geometry '{0}' config can not be blank or error type: '{1}'.", name, type(self._config_))
        for key, value in self.__rules__.items():
            assert_util.is_true(key in self._config_.keys(),
                                "'{0}' is not exist in geometry '{1}' config.", key, name)

            config_value = self._config_[key]
            assert_util.same_type(type(config_value), eval(value),
                                  "'{0}' of geometry '{1}' config is not the type: '{2}'.", config_value, name, value)

    def set_config(self,
                   config: dict) -> None:
        """
        Set geometry configuration. And transform list to tuple
        """
        for key, value in config.items():
            value = tuple(value) if isinstance(value, list) else value
            self._config_[key] = value

    def load_base_config(self) -> None:
        self.point_dithering = self._config_["point_dithering"]
        self.move_plane_range = self._config_['move_plane_range']
        self.move_y_range = self._config_['move_y_range']
        self.plane_rotate_degree_range = self._config_["plane_rotate_degree_range"]
        self.density = self._config_["density"]
        self.equinox_range = self._config_["equinox_range"]

    @abstractmethod
    def load_config(self) -> None:
        """
        Load config
        """
        pass

    def get_config(self) -> dict[str, Any]:
        """
        Get geometry configuration
        """
        return self._config_

    def __load_rules__(self):
        """
        Load geometry rules
        """
        name = self.name()
        assert_util.is_true(name is not None and len(name) > 0, "geometry name should not be blank.")

        assert_util.is_true(os.path.exists(self.__RULES_PATH__), "rules file: '{0}' is not exist.", self.__RULES_PATH__)
        geometries_rules = toml.load(self.__RULES_PATH__)
        assert_util.is_true(name in geometries_rules.keys() and isinstance(geometries_rules[name], dict),
                            "geometry rules of '{0}' is not exist.", name)

        rules = geometries_rules[name]
        supported_type = ("int", "float", "tuple")
        for key, value in rules.items():
            assert_util.is_true(value in supported_type,
                                "geometry rule '{0}' has not support the type: {1}.", key, value)
        self.__rules__ = rules

    @abstractmethod
    def generate_strokes(self) -> Strokes:
        """
        Generate strokes
        """
        pass


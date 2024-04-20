from enum import Enum


class BuiltinGeometry(Enum):
    """
    Builtin geometry enum.
    """

    Cone = "cone"
    Cuboid = "cuboid"
    Cylinder = "cylinder"
    Freeform = "freeform"
    Hemisphere = "hemisphere"
    Hip = "hip"
    Platform = "platform"
    Pyramid = "pyramid"
    Shed = "shed"

    def get_name(self) -> str:
        """
        Get builtin geometry name.
        :return: Builtin geometry name.
        """
        return self.value

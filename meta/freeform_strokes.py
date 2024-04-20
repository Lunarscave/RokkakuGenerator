from typing_extensions import Self

from meta import Strokes
from numpy import ndarray

from utils import assert_util


class FreeformStrokes(Strokes):
    """
    Freeform strokes based on strokes. Add one attribute: types.
    """

    def __init__(self):
        """
        Init freeform strokes.
        """
        super().__init__()
        self.types: ndarray[int] | None = None

    @classmethod
    def load_points(
            cls,
            *points: ndarray[ndarray] | ndarray
    ) -> Self:
        """
        Load strokes value.
        :param points: Freeform stroke points.
        :return: Freeform stroke instance.
        """
        strokes = FreeformStrokes()
        strokes.set_value(cls.mark_points3d(points[0], 1))
        strokes.append_points(*points[1:])
        return strokes

    def load_plane_strokes_types(
            self,
            types: ndarray[int] | ndarray
    ) -> Self:
        """
        Load strokes types.
        :param types: Freeform stroke types.
        :return: Freeform stroke instance.
        """
        assert_util.is_not_none(self.value, "Freeform strokes could not be blank.")
        self.types = types
        return self

    def get_types(self) -> ndarray[int]:
        """
        Get freeform strokes types.
        :return: Strokes types.
        """
        return self.types

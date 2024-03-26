from meta import Strokes
from numpy import ndarray

from utils import assert_util


class FreeformStrokes(Strokes):
    def __init__(self):
        super().__init__()
        self.types = None

    @classmethod
    def load_points(cls, *points: ndarray):
        """
        Load strokes value
        """
        strokes = FreeformStrokes()
        strokes.set_value(cls.mark_points3d(points[0], 0))
        strokes.append_points(*points[1:])
        return strokes

    def load_plane_strokes_types(self, types: ndarray):
        assert_util.is_not_none(self.value, "Freeform strokes could not be blank.")
        self.types = types
        return self

    def get_types(self) -> ndarray:
        return self.types

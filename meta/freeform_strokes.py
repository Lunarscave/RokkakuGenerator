from meta import Strokes
from numpy import ndarray

from utils import assert_util


class FreeformStrokes(Strokes):
    def __init__(self):
        super().__init__()
        self.types = None

    def load_plane_strokes_types(self, types: ndarray):
        assert_util.is_not_none(self.value, "Freeform strokes could not be blank.")
        assert_util.is_true(len(self.value) == len(types),
                            "Freeform strokes types length and strokes length are not equal.")
        self.types = types
        return self

    def get_types(self) -> ndarray:
        return self.types

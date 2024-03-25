from typing import Any


def is_true(expression: bool,
            error_message: str,
            *values: Any) -> None:
    """
    assert the expression is true, or throw ValueError with error message(format values)
    """
    if not expression:
        raise ValueError(error_message.format(*values))


def is_false(expression: bool,
             error_message: str,
             *values: Any) -> None:
    """
    assert the expression is false, or throw ValueError with error message(format values)
    """
    if expression:
        raise ValueError(error_message.format(*values))


def is_none(instance: Any,
            error_message: str,
            *values: Any) -> None:
    """
    assert the instance is none, or throw ValueError with error message(format values)
    """
    if instance is not None:
        raise ValueError(error_message.format(*values))


def is_not_none(instance: Any,
                error_message: str,
                *values: Any) -> None:
    """
    assert the instance is not none, or throw ValueError with error message(format values)
    """
    if instance is None:
        raise ValueError(error_message.format(*values))


def same_type(type1: type,
              type2: type,
              error_message: str,
              *values: Any) -> None:
    """
    assert the expression is true, or throw ValueError with error message(format values)
    """
    if isinstance(type1, type2):
        raise ValueError(error_message.format(*values))

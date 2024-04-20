from typing import Any


def is_true(
        expression: bool,
        error_message: str,
        *values: Any
) -> None:
    """
    Assert the expression is true, or throw ValueError with error message(format values).
    :param expression: Validate expression.
    :param error_message: Error message.
    :param values: Values of error message.
    """
    if not expression:
        raise ValueError(error_message.format(*values))


def is_false(
        expression: bool,
        error_message: str,
        *values: Any
) -> None:
    """
    Assert the expression is false, or throw ValueError with error message(format values).
    :param expression: Validate expression.
    :param error_message: Error message.
    :param values: Values of error message.
    """
    if expression:
        raise ValueError(error_message.format(*values))


def is_none(
        instance: Any,
        error_message: str,
        *values: Any
) -> None:
    """
    Assert the instance is none, or throw ValueError with error message(format values).
    :param instance: Validate instance.
    :param error_message: Error message.
    :param values: Values of error message.
    """
    if instance is not None:
        raise ValueError(error_message.format(*values))


def is_not_none(
        instance: Any,
        error_message: str,
        *values: Any
) -> None:
    """
    Assert the instance is not none, or throw ValueError with error message(format values).
    :param instance: Validate instance.
    :param error_message: Error message.
    :param values: Values of error message.
    """
    if instance is None:
        raise ValueError(error_message.format(*values))


def same_type(
        type1: type,
        type2: type,
        error_message: str,
        *values: Any
) -> None:
    """
    Assert the expression is true, or throw ValueError with error message(format values).
    :param type1: Validation type1.
    :param type2: Validation type2.
    :param error_message: Error message.
    :param values: Values of error message.
    """
    if isinstance(type1, type2):
        raise ValueError(error_message.format(*values))

import os.path

import h5py
from h5py import File
from numpy import ndarray
from utils import assert_util


def validate_file_path(file_path: str) -> None:
    """
    Validate the file path exist and is file and file matches *.hdf5
    :param file_path: Hdf5 file path.
    """
    file_path = os.path.abspath(file_path)
    directory = os.path.dirname(file_path)
    assert_util.is_true(os.path.exists(directory), "could not find the file path.")
    assert_util.is_true(os.path.splitext(file_path)[-1] == ".hdf5", "file path extension does not matches the .hdf5")


def validate_directory_path(directory_path: str) -> None:
    """
    Validate the directory path exist and is directory
    :param directory_path: Hdf5 directory path.
    """
    directory_path = os.path.abspath(directory_path)
    directory = os.path.dirname(directory_path)
    assert_util.is_true(os.path.exists(directory), "could not find the directory path.")
    assert_util.is_true(os.path.splitext(directory_path)[-1] == "", "path is not a directory")


def save_file(
        file_path: str,
        datas: list,
        data_names: list[str],
        dtypes: list[str]
) -> None:
    """
    Save hdf5 file.
    :param file_path: Hdf5 file path.
    :param datas: Hdf5 datas (of list).
    :param data_names: Hdf5 data names (of list).
    :param dtypes: Hdf5 dtypes (of list).
    """
    validate_file_path(file_path)
    h5 = h5py.File(file_path, "w")
    for i, _ in enumerate(datas):
        h5.create_dataset(data_names[i], data=datas[i], compression='gzip', compression_opts=4, dtype=dtypes[i])
    h5.close()


def save_files(
        directory_path: str,
        datas: list[list],
        data_names: list[list[str]],
        dtypes: list[list[str]],
        file_names: list[str] = None
) -> None:
    """
    Save hdf5 files.
    :param directory_path: Hdf5 directory path.
    :param datas: Hdf5 datas (of list).
    :param data_names: Hdf5 data names (of list).
    :param dtypes: Hdf5 dtypes (of list).
    :param file_names: Hdf5 file names (of list).
    """
    validate_directory_path(directory_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    for i, _ in enumerate(data_names):
        file_path = f"{directory_path}\\{i + 1 if file_names is None else file_names[i]}.hdf5"
        save_file(file_path, datas[i], data_names[i], dtypes[i])


def read_file(file_path: str) -> File:
    """
    Read hdf5 file
    :param file_path: Hdf5 file path.
    :return: Hdf5 file.
    """
    validate_file_path(file_path)
    return h5py.File(file_path, "r")


def get_data_2ndarray(
        file: File,
        data_name: str
) -> ndarray:
    """
    Get data_value of file.
    :param file: Hdf5 file.
    :param data_name: Data name of hdf5 file.
    :return: Data instance.
    """
    return file.get(data_name)

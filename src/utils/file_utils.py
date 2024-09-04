# src/utils/file_utils.py

import os
import shutil
import requests
from pathlib import Path
from tqdm.auto import tqdm


def ensure_dir_exists(directory):
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def download_file(url, output_path, chunk_size=1024):
    """
    Download a file from the specified URL and save it to the output path.

    Args:
        url (str): The URL of the file to download.
        output_path (str): Path to save the downloaded file.
        chunk_size (int): Chunk size for downloading the file.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    output_dir = os.path.dirname(output_path)

    ensure_dir_exists(output_dir)

    with open(output_path, "wb") as file, tqdm(
            desc=f"Downloading {os.path.basename(output_path)}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            file.write(data)
            bar.update(len(data))


def list_files(directory, extension=None):
    """
    List all files in a directory, optionally filtering by file extension.

    Args:
        directory (str): Path to the directory.
        extension (str, optional): File extension to filter by (e.g., '.txt').

    Returns:
        list: List of file paths.
    """
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if extension is None or filename.endswith(extension):
                files.append(os.path.join(root, filename))
    return files


def move_file(src_path, dest_path):
    """
    Move a file from source path to destination path.

    Args:
        src_path (str): Path to the source file.
        dest_path (str): Path to the destination file.
    """
    ensure_dir_exists(os.path.dirname(dest_path))
    shutil.move(src_path, dest_path)


def copy_file(src_path, dest_path):
    """
    Copy a file from source path to destination path.

    Args:
        src_path (str): Path to the source file.
        dest_path (str): Path to the destination file.
    """
    ensure_dir_exists(os.path.dirname(dest_path))
    shutil.copy(src_path, dest_path)


def delete_file(file_path):
    """
    Delete the specified file.

    Args:
        file_path (str): Path to the file to be deleted.
    """
    if os.path.exists(file_path):
        os.remove(file_path)


def get_file_size(file_path):
    """
    Get the size of a file in bytes.

    Args:
        file_path (str): Path to the file.

    Returns:
        int: Size of the file in bytes.
    """
    return os.path.getsize(file_path)


def is_file_empty(file_path):
    """
    Check if a file is empty.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if the file is empty, False otherwise.
    """
    return os.path.getsize(file_path) == 0


def read_file(file_path, mode="r"):
    """
    Read the contents of a file.

    Args:
        file_path (str): Path to the file.
        mode (str): Mode to open the file (default is "r" for reading).

    Returns:
        str: Contents of the file.
    """
    with open(file_path, mode) as file:
        return file.read()


def write_file(file_path, content, mode="w"):
    """
    Write content to a file.

    Args:
        file_path (str): Path to the file.
        content (str): Content to write to the file.
        mode (str): Mode to open the file (default is "w" for writing).
    """
    ensure_dir_exists(os.path.dirname(file_path))
    with open(file_path, mode) as file:
        file.write(content)

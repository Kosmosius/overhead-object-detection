# tests/unit/src/utils/test_file_utils.py

import os
import pytest
import shutil
from src.utils.file_utils import ensure_dir_exists, download_file, list_files, move_file, copy_file, delete_file, get_file_size, is_file_empty, read_file, write_file

@pytest.fixture
def tmp_directory(tmp_path):
    """Fixture to create a temporary directory for file operations."""
    return str(tmp_path)

@pytest.fixture
def test_file(tmp_directory):
    """Fixture to create a temporary test file."""
    file_path = os.path.join(tmp_directory, "test_file.txt")
    with open(file_path, "w") as f:
        f.write("This is a test.")
    return file_path

def test_ensure_dir_exists(tmp_directory):
    dir_path = os.path.join(tmp_directory, "new_folder")
    ensure_dir_exists(dir_path)
    assert os.path.exists(dir_path)

def test_list_files(tmp_directory, test_file):
    files = list_files(tmp_directory)
    assert test_file in files

def test_list_files_with_extension(tmp_directory, test_file):
    files = list_files(tmp_directory, extension=".txt")
    assert test_file in files

def test_move_file(tmp_directory, test_file):
    dest_path = os.path.join(tmp_directory, "new_file.txt")
    move_file(test_file, dest_path)
    assert os.path.exists(dest_path)
    assert not os.path.exists(test_file)

def test_copy_file(tmp_directory, test_file):
    dest_path = os.path.join(tmp_directory, "copy_file.txt")
    copy_file(test_file, dest_path)
    assert os.path.exists(dest_path)
    assert os.path.exists(test_file)

def test_delete_file(test_file):
    delete_file(test_file)
    assert not os.path.exists(test_file)

def test_get_file_size(test_file):
    file_size = get_file_size(test_file)
    assert file_size > 0

def test_is_file_empty(test_file):
    assert not is_file_empty(test_file)

def test_read_file(test_file):
    content = read_file(test_file)
    assert content == "This is a test."

def test_write_file(tmp_directory):
    file_path = os.path.join(tmp_directory, "write_test.txt")
    write_file(file_path, "Hello, World!")
    assert os.path.exists(file_path)
    content = read_file(file_path)
    assert content == "Hello, World!"

def test_download_file(tmp_directory, requests_mock):
    url = "http://example.com/testfile"
    file_path = os.path.join(tmp_directory, "downloaded_file.txt")

    # Mock the download response
    requests_mock.get(url, text="Downloaded content")
    download_file(url, file_path)

    assert os.path.exists(file_path)
    content = read_file(file_path)
    assert content == "Downloaded content"

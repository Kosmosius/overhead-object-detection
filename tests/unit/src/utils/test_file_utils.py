# tests/unit/src/utils/test_file_utils.py

import os
import pytest
import shutil
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.utils.file_utils import (
    ensure_dir_exists,
    download_file,
    parallel_download_files,
    list_files,
    move_file,
    copy_file,
    delete_file,
    get_file_size,
    is_file_empty,
    read_file,
    write_file
)
from requests.exceptions import RequestException

# --- Fixtures ---

@pytest.fixture
def temp_dir():
    """Fixture to create and clean up a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def temp_file(temp_dir):
    """Fixture to create a temporary file."""
    file_path = os.path.join(temp_dir, "test_file.txt")
    with open(file_path, "w") as f:
        f.write("This is a test.")
    return file_path

@pytest.fixture
def empty_file(temp_dir):
    """Fixture to create an empty file."""
    file_path = os.path.join(temp_dir, "empty_file.txt")
    open(file_path, "w").close()
    return file_path

@pytest.fixture
def mocked_requests_get_success():
    """Fixture to mock successful HTTP GET requests."""
    with patch("src.utils.file_utils.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"content-length": "20"}
        mock_response.iter_content = MagicMock(return_value=[b'a'*10, b'b'*10])
        mock_get.return_value = mock_response
        yield mock_get

@pytest.fixture
def mocked_requests_get_failure():
    """Fixture to mock failed HTTP GET requests."""
    with patch("src.utils.file_utils.requests.get") as mock_get:
        mock_get.side_effect = RequestException("Failed to connect")
        yield mock_get

@pytest.fixture
def mocked_requests_get_partial_failure():
    """Fixture to mock partial failures in HTTP GET requests."""
    with patch("src.utils.file_utils.requests.get") as mock_get:
        # First two attempts fail, third succeeds
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"content-length": "20"}
        mock_response.iter_content = MagicMock(return_value=[b'a'*10, b'b'*10])

        mock_get.side_effect = [RequestException("Failed to connect"),
                                RequestException("Failed to connect"),
                                mock_response]
        yield mock_get

# --- Test Cases ---

# 1. Directory Management Tests

def test_ensure_dir_exists_creates_directory_when_not_exists(temp_dir):
    """Test that ensure_dir_exists creates the directory if it does not exist."""
    new_dir = os.path.join(temp_dir, "new_folder")
    ensure_dir_exists(new_dir)
    assert os.path.isdir(new_dir), "Directory was not created."

def test_ensure_dir_exists_no_action_if_exists(temp_dir):
    """Test that ensure_dir_exists does nothing if the directory already exists."""
    existing_dir = os.path.join(temp_dir, "existing_folder")
    os.makedirs(existing_dir)
    with patch("src.utils.file_utils.Path.mkdir") as mock_mkdir:
        ensure_dir_exists(existing_dir)
        mock_mkdir.assert_not_called(), "Path.mkdir should not be called if directory exists."

def test_ensure_dir_exists_invalid_path():
    """Test that ensure_dir_exists raises an error with an invalid path."""
    invalid_path = "/invalid_path/\0"
    with pytest.raises(OSError):
        ensure_dir_exists(invalid_path)

# 2. File Download Tests

def test_download_file_success(mocked_requests_get_success, temp_dir):
    """Test successful file download."""
    url = "http://example.com/testfile"
    output_path = os.path.join(temp_dir, "downloaded_file.txt")
    
    download_file(url, output_path)
    
    assert os.path.exists(output_path), "Downloaded file does not exist."
    with open(output_path, "rb") as f:
        content = f.read()
    assert content == b'a'*10 + b'b'*10, "Downloaded file content mismatch."

def test_download_file_failure_with_retries(mocked_requests_get_failure, temp_dir):
    """Test that download_file retries upon failure and eventually raises an error."""
    url = "http://example.com/testfile"
    output_path = os.path.join(temp_dir, "downloaded_file.txt")
    
    with pytest.raises(RequestException):
        download_file(url, output_path, retries=2, backoff=1)
    
    assert not os.path.exists(output_path), "File should not exist after failed downloads."

def test_download_file_partial_failure_then_success(mocked_requests_get_partial_failure, temp_dir):
    """Test that download_file retries upon initial failures and succeeds on subsequent attempts."""
    url = "http://example.com/testfile"
    output_path = os.path.join(temp_dir, "downloaded_file.txt")
    
    download_file(url, output_path, retries=3, backoff=1)
    
    assert os.path.exists(output_path), "Downloaded file does not exist after retries."
    with open(output_path, "rb") as f:
        content = f.read()
    assert content == b'a'*10 + b'b'*10, "Downloaded file content mismatch after retries."

def test_download_file_no_content_length(mocked_requests_get_success, temp_dir):
    """Test downloading a file when Content-Length header is missing."""
    url = "http://example.com/testfile_no_length"
    output_path = os.path.join(temp_dir, "downloaded_no_length.txt")
    
    # Modify the mock to remove content-length
    mocked_requests_get_success.return_value.headers = {}
    
    download_file(url, output_path)
    
    assert os.path.exists(output_path), "Downloaded file does not exist."
    with open(output_path, "rb") as f:
        content = f.read()
    assert content == b'a'*10 + b'b'*10, "Downloaded file content mismatch without Content-Length."

def test_download_file_invalid_url(mocked_requests_get_failure, temp_dir):
    """Test that download_file raises an error with an invalid URL."""
    url = "htp://invalid_url"
    output_path = os.path.join(temp_dir, "invalid_download.txt")
    
    with pytest.raises(RequestException):
        download_file(url, output_path, retries=1, backoff=1)
    
    assert not os.path.exists(output_path), "File should not exist after invalid URL download attempt."

def test_parallel_download_files_success(mocked_requests_get_success, temp_dir):
    """Test successful parallel downloading of multiple files."""
    urls = [
        "http://example.com/file1.txt",
        "http://example.com/file2.txt",
        "http://example.com/file3.txt"
    ]
    output_dir = os.path.join(temp_dir, "downloads")
    
    download_content = b'content1' + b'content2' + b'content3'
    
    with patch("src.utils.file_utils.download_file") as mock_download_file:
        mock_download_file.side_effect = lambda url, path, *args, **kwargs: open(path, "wb").write(b'data')
        parallel_download_files(urls, output_dir, max_workers=3)
    
    for url in urls:
        filename = os.path.basename(url)
        file_path = os.path.join(output_dir, filename)
        assert os.path.exists(file_path), f"File {filename} was not downloaded."

def test_parallel_download_files_partial_failures(temp_dir):
    """Test parallel_download_files handles partial download failures gracefully."""
    urls = [
        "http://example.com/file1.txt",
        "http://example.com/file2.txt",
        "http://example.com/file3.txt"
    ]
    output_dir = os.path.join(temp_dir, "downloads")
    
    with patch("src.utils.file_utils.download_file") as mock_download_file:
        def side_effect(url, path, *args, **kwargs):
            if "file2.txt" in url:
                raise RequestException("Download failed for file2.txt")
            else:
                with open(path, "wb") as f:
                    f.write(b'data')
        mock_download_file.side_effect = side_effect
        
        parallel_download_files(urls, output_dir, max_workers=3)
    
    # Check that file1 and file3 are downloaded, file2 is not
    assert os.path.exists(os.path.join(output_dir, "file1.txt")), "file1.txt was not downloaded."
    assert not os.path.exists(os.path.join(output_dir, "file2.txt")), "file2.txt should not be downloaded due to failure."
    assert os.path.exists(os.path.join(output_dir, "file3.txt")), "file3.txt was not downloaded."

def test_parallel_download_files_empty_url_list(temp_dir):
    """Test that parallel_download_files handles an empty URL list gracefully."""
    urls = []
    output_dir = os.path.join(temp_dir, "downloads")
    
    with patch("src.utils.file_utils.download_file") as mock_download_file:
        parallel_download_files(urls, output_dir, max_workers=3)
    
    assert os.path.exists(output_dir), "Output directory was not created."
    assert len(os.listdir(output_dir)) == 0, "Output directory should be empty."

# 3. File Listing Tests

def test_list_files_all(temp_dir, temp_file):
    """Test that list_files returns all files in the directory."""
    files = list_files(temp_dir)
    assert temp_file in files, "list_files did not return the existing file."

def test_list_files_with_extension(temp_dir, temp_file):
    """Test that list_files filters files by the specified extension."""
    files = list_files(temp_dir, extension=".txt")
    assert temp_file in files, "list_files did not return the file with the specified extension."
    assert len(files) == 1, "list_files returned incorrect number of files."

def test_list_files_nonexistent_directory():
    """Test that list_files raises FileNotFoundError for non-existent directories."""
    nonexistent_dir = "/path/to/nonexistent/directory"
    with pytest.raises(FileNotFoundError):
        list_files(nonexistent_dir)

def test_list_files_empty_directory(temp_dir):
    """Test that list_files returns an empty list for empty directories."""
    files = list_files(temp_dir)
    assert files == [], "list_files should return an empty list for empty directories."

# 4. File Movement and Copying Tests

def test_move_file_success(temp_dir, temp_file):
    """Test that move_file successfully moves a file."""
    dest_path = os.path.join(temp_dir, "moved_file.txt")
    move_file(temp_file, dest_path)
    assert not os.path.exists(temp_file), "Source file was not moved."
    assert os.path.exists(dest_path), "Destination file does not exist."

def test_move_file_nonexistent_source(temp_dir):
    """Test that move_file raises FileNotFoundError when source file does not exist."""
    src_path = os.path.join(temp_dir, "nonexistent.txt")
    dest_path = os.path.join(temp_dir, "dest.txt")
    with pytest.raises(FileNotFoundError):
        move_file(src_path, dest_path)

def test_move_file_invalid_destination(temp_dir):
    """Test that move_file raises an error when destination path is invalid."""
    src_path = os.path.join(temp_dir, "test_move.txt")
    with open(src_path, "w") as f:
        f.write("Test")
    dest_path = "/invalid_path/dest.txt"
    with pytest.raises(OSError):
        move_file(src_path, dest_path)

def test_copy_file_success(temp_dir, temp_file):
    """Test that copy_file successfully copies a file."""
    dest_path = os.path.join(temp_dir, "copied_file.txt")
    copy_file(temp_file, dest_path)
    assert os.path.exists(temp_file), "Source file was unexpectedly moved."
    assert os.path.exists(dest_path), "Destination file does not exist."

def test_copy_file_nonexistent_source(temp_dir):
    """Test that copy_file raises FileNotFoundError when source file does not exist."""
    src_path = os.path.join(temp_dir, "nonexistent.txt")
    dest_path = os.path.join(temp_dir, "dest.txt")
    with pytest.raises(FileNotFoundError):
        copy_file(src_path, dest_path)

def test_copy_file_invalid_destination(temp_dir):
    """Test that copy_file raises an error when destination path is invalid."""
    src_path = os.path.join(temp_dir, "test_copy.txt")
    with open(src_path, "w") as f:
        f.write("Test")
    dest_path = "/invalid_path/dest.txt"
    with pytest.raises(OSError):
        copy_file(src_path, dest_path)

# 5. File Deletion Tests

def test_delete_file_success(temp_dir, temp_file):
    """Test that delete_file successfully deletes a file."""
    delete_file(temp_file)
    assert not os.path.exists(temp_file), "File was not deleted."

def test_delete_file_nonexistent(temp_dir):
    """Test that delete_file raises FileNotFoundError when file does not exist."""
    file_path = os.path.join(temp_dir, "nonexistent.txt")
    with pytest.raises(FileNotFoundError):
        delete_file(file_path)

# 6. File Size and Emptiness Tests

def test_get_file_size_non_empty_file(temp_file):
    """Test that get_file_size returns the correct size for a non-empty file."""
    size = get_file_size(temp_file)
    assert size == len("This is a test."), "File size does not match expected value."

def test_get_file_size_nonexistent_file(temp_dir):
    """Test that get_file_size raises FileNotFoundError for non-existent files."""
    file_path = os.path.join(temp_dir, "nonexistent.txt")
    with pytest.raises(FileNotFoundError):
        get_file_size(file_path)

def test_is_file_empty_empty_file(empty_file):
    """Test that is_file_empty returns True for an empty file."""
    assert is_file_empty(empty_file), "is_file_empty should return True for empty files."

def test_is_file_empty_non_empty_file(temp_file):
    """Test that is_file_empty returns False for a non-empty file."""
    assert not is_file_empty(temp_file), "is_file_empty should return False for non-empty files."

def test_is_file_empty_nonexistent_file(temp_dir):
    """Test that is_file_empty raises FileNotFoundError for non-existent files."""
    file_path = os.path.join(temp_dir, "nonexistent.txt")
    with pytest.raises(FileNotFoundError):
        is_file_empty(file_path)

# 7. File Read and Write Tests

def test_read_file_success(temp_file):
    """Test that read_file successfully reads the contents of a file."""
    content = read_file(temp_file)
    assert content == "This is a test.", "read_file returned incorrect content."

def test_read_file_nonexistent(temp_dir):
    """Test that read_file raises FileNotFoundError when file does not exist."""
    file_path = os.path.join(temp_dir, "nonexistent.txt")
    with pytest.raises(FileNotFoundError):
        read_file(file_path)

def test_write_file_success(temp_dir):
    """Test that write_file successfully writes content to a file."""
    file_path = os.path.join(temp_dir, "write_test.txt")
    content = "Hello, World!"
    write_file(file_path, content)
    assert os.path.exists(file_path), "write_file did not create the file."
    with open(file_path, "r") as f:
        read_content = f.read()
    assert read_content == content, "write_file wrote incorrect content."

def test_write_file_overwrite(temp_dir, temp_file):
    """Test that write_file successfully overwrites an existing file."""
    content = "Overwritten content."
    write_file(temp_file, content)
    with open(temp_file, "r") as f:
        read_content = f.read()
    assert read_content == content, "write_file did not overwrite the file correctly."

def test_write_file_invalid_path():
    """Test that write_file raises an error when destination path is invalid."""
    file_path = "/invalid_path/write_test.txt"
    content = "Test content."
    with pytest.raises(OSError):
        write_file(file_path, content)

# 8. Edge Case Tests

def test_download_file_invalid_url_format(temp_dir):
    """Test that download_file raises an error with an invalid URL format."""
    url = "htp:/invalid-url"
    output_path = os.path.join(temp_dir, "invalid_url.txt")
    with pytest.raises(RequestException):
        download_file(url, output_path, retries=1, backoff=1)
    assert not os.path.exists(output_path), "File should not exist after invalid URL download attempt."

def test_download_file_extremely_large_file(mocked_requests_get_success, temp_dir):
    """Test that download_file can handle extremely large files."""
    url = "http://example.com/largefile"
    output_path = os.path.join(temp_dir, "large_file.txt")
    
    # Mock the iter_content to simulate a large file
    with patch("src.utils.file_utils.tqdm") as mock_tqdm:
        mock_tqdm.return_value.__enter__.return_value = MagicMock()
        download_file(url, output_path, chunk_size=1024, retries=3, backoff=1)
    
    assert os.path.exists(output_path), "Large file was not downloaded."
    # Assuming 'a'*10 + 'b'*10 as per mocked_requests_get_success
    with open(output_path, "rb") as f:
        content = f.read()
    assert content == b'a'*10 + b'b'*10, "Large file content mismatch."

def test_parallel_download_files_extremely_large_number_of_files(temp_dir):
    """Test that parallel_download_files can handle downloading a large number of files."""
    urls = [f"http://example.com/file{i}.txt" for i in range(100)]
    output_dir = os.path.join(temp_dir, "downloads")
    
    with patch("src.utils.file_utils.download_file") as mock_download_file:
        mock_download_file.side_effect = lambda url, path, *args, **kwargs: open(path, "wb").write(b'data')
        parallel_download_files(urls, output_dir, max_workers=10)
    
    for url in urls:
        filename = os.path.basename(url)
        file_path = os.path.join(output_dir, filename)
        assert os.path.exists(file_path), f"File {filename} was not downloaded."

def test_move_file_permissions_error(temp_dir):
    """Test that move_file raises PermissionError when lacking permissions."""
    src_path = os.path.join(temp_dir, "test_move.txt")
    dest_path = os.path.join(temp_dir, "dest_move.txt")
    with open(src_path, "w") as f:
        f.write("Test")
    
    with patch("src.utils.file_utils.shutil.move", side_effect=PermissionError("No permission to move file")):
        with pytest.raises(PermissionError):
            move_file(src_path, dest_path)
    
    assert os.path.exists(src_path), "Source file should still exist after failed move."

def test_copy_file_permissions_error(temp_dir):
    """Test that copy_file raises PermissionError when lacking permissions."""
    src_path = os.path.join(temp_dir, "test_copy.txt")
    dest_path = os.path.join(temp_dir, "dest_copy.txt")
    with open(src_path, "w") as f:
        f.write("Test")
    
    with patch("src.utils.file_utils.shutil.copy", side_effect=PermissionError("No permission to copy file")):
        with pytest.raises(PermissionError):
            copy_file(src_path, dest_path)
    
    assert not os.path.exists(dest_path), "Destination file should not exist after failed copy."

# 9. File Read and Write with Different Modes

def test_read_file_binary_mode(temp_dir):
    """Test that read_file can read files in binary mode."""
    file_path = os.path.join(temp_dir, "binary_file.bin")
    binary_content = b'\x00\xFF\x00\xFF'
    with open(file_path, "wb") as f:
        f.write(binary_content)
    
    content = read_file(file_path, mode="rb")
    assert content == binary_content, "read_file did not return correct binary content."

def test_write_file_append_mode(temp_dir):
    """Test that write_file can append content to a file."""
    file_path = os.path.join(temp_dir, "append_test.txt")
    initial_content = "Initial content.\n"
    append_content = "Appended content.\n"
    
    write_file(file_path, initial_content)
    write_file(file_path, append_content, mode="a")
    
    with open(file_path, "r") as f:
        content = f.read()
    
    assert content == initial_content + append_content, "write_file did not append content correctly."

# 10. File Deletion Tests with Permissions

def test_delete_file_permissions_error(temp_dir):
    """Test that delete_file raises PermissionError when lacking permissions."""
    file_path = os.path.join(temp_dir, "test_delete.txt")
    with open(file_path, "w") as f:
        f.write("Test")
    
    # Make the file read-only
    os.chmod(file_path, 0o444)
    
    with pytest.raises(PermissionError):
        delete_file(file_path)
    
    assert os.path.exists(file_path), "File should still exist after failed deletion."

# 11. Edge Case: File Paths with Special Characters

def test_file_operations_with_special_characters(temp_dir):
    """Test file operations with file paths containing special characters."""
    filename = "spécial_файл.txt"
    file_path = os.path.join(temp_dir, filename)
    
    write_file(file_path, "Special characters content.")
    assert os.path.exists(file_path), "File with special characters was not created."
    
    content = read_file(file_path)
    assert content == "Special characters content.", "Content mismatch for file with special characters."
    
    copied_path = os.path.join(temp_dir, "copy_spécial_файл.txt")
    copy_file(file_path, copied_path)
    assert os.path.exists(copied_path), "Copied file with special characters does not exist."
    
    delete_file(copied_path)
    assert not os.path.exists(copied_path), "Copied file with special characters was not deleted."

# 12. Edge Case: Extremely Large Chunk Size in Download

def test_download_file_extremely_large_chunk_size(mocked_requests_get_success, temp_dir):
    """Test download_file with an extremely large chunk size."""
    url = "http://example.com/largechunkfile"
    output_path = os.path.join(temp_dir, "large_chunk_file.txt")
    
    download_file(url, output_path, chunk_size=10**6)  # 1MB chunk size
    
    assert os.path.exists(output_path), "Downloaded file does not exist."
    with open(output_path, "rb") as f:
        content = f.read()
    assert content == b'a'*10 + b'b'*10, "Downloaded file content mismatch with large chunk size."

# 13. Edge Case: Download to Existing File

def test_download_file_to_existing_file(mocked_requests_get_success, temp_dir):
    """Test that download_file overwrites an existing file."""
    url = "http://example.com/testfile"
    output_path = os.path.join(temp_dir, "downloaded_file.txt")
    
    # Create an existing file
    with open(output_path, "wb") as f:
        f.write(b'old_data')
    
    download_file(url, output_path)
    
    with open(output_path, "rb") as f:
        content = f.read()
    assert content == b'a'*10 + b'b'*10, "download_file did not overwrite the existing file correctly."

# 14. Edge Case: Read and Write Binary Files

def test_read_write_binary_file(temp_dir):
    """Test reading and writing binary files."""
    file_path = os.path.join(temp_dir, "binary_test.bin")
    binary_content = b'\x00\xFF\x00\xFF'
    
    write_file(file_path, binary_content, mode="wb")
    read_content = read_file(file_path, mode="rb")
    assert read_content == binary_content, "Binary content mismatch after write and read."

# 15. Edge Case: Download File with Redirects

def test_download_file_with_redirects(temp_dir):
    """Test download_file handles HTTP redirects."""
    url = "http://example.com/redirect"
    output_path = os.path.join(temp_dir, "redirected_file.txt")
    
    with patch("src.utils.file_utils.requests.get") as mock_get:
        mock_response_redirect = MagicMock()
        mock_response_redirect.raise_for_status.side_effect = RequestException("Redirect not handled")
        mock_get.return_value = mock_response_redirect
        
        with pytest.raises(RequestException):
            download_file(url, output_path, retries=1, backoff=1)
    
    assert not os.path.exists(output_path), "File should not exist after failed redirect handling."

# 16. Edge Case: List Files with No Extension

def test_list_files_no_extension(temp_dir):
    """Test that list_files can list files without any extension."""
    file_path = os.path.join(temp_dir, "no_extension")
    with open(file_path, "w") as f:
        f.write("No extension file.")
    
    files = list_files(temp_dir)
    assert file_path in files, "File without extension was not listed."

# 17. Edge Case: Write File to Nested Directory

def test_write_file_nested_directory(temp_dir):
    """Test that write_file can create nested directories when writing a file."""
    nested_dir = os.path.join(temp_dir, "nested", "dir", "structure")
    file_path = os.path.join(nested_dir, "nested_file.txt")
    content = "Nested directory file content."
    
    write_file(file_path, content)
    assert os.path.exists(file_path), "File in nested directories was not created."
    
    with open(file_path, "r") as f:
        read_content = f.read()
    assert read_content == content, "Content mismatch for file in nested directories."

# 18. Edge Case: Read File with Incorrect Mode

def test_read_file_incorrect_mode(temp_dir):
    """Test that read_file raises an error when opened in incorrect mode."""
    file_path = os.path.join(temp_dir, "test_mode.txt")
    with open(file_path, "w") as f:
        f.write("Test content.")
    
    with pytest.raises(ValueError):
        read_file(file_path, mode="invalid_mode")

# 19. Edge Case: Write File with Invalid Mode

def test_write_file_invalid_mode(temp_dir):
    """Test that write_file raises an error when opened with an invalid mode."""
    file_path = os.path.join(temp_dir, "test_write_invalid_mode.txt")
    content = "Test content."
    
    with pytest.raises(ValueError):
        write_file(file_path, content, mode="invalid_mode")

# 20. Edge Case: Delete File Already Deleted

def test_delete_file_already_deleted(temp_dir, temp_file):
    """Test that delete_file raises FileNotFoundError when attempting to delete an already deleted file."""
    delete_file(temp_file)
    assert not os.path.exists(temp_file), "File was not deleted."
    
    with pytest.raises(FileNotFoundError):
        delete_file(temp_file)


# src/utils/file_utils.py

import os
import shutil
import requests
import logging
from pathlib import Path
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from time import sleep
from requests.exceptions import RequestException

# Set up logging for the file utilities module
logger = logging.getLogger(__name__)

def ensure_dir_exists(directory: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory (str): Path to the directory.
    """
    path = Path(directory)
    if not path.exists():
        logger.info(f"Creating directory: {directory}")
        path.mkdir(parents=True, exist_ok=True)

def download_file(url: str, output_path: str, chunk_size: int = 1024, retries: int = 3, backoff: int = 2) -> None:
    """
    Download a file from the specified URL and save it to the output path, with retry mechanism.

    Args:
        url (str): The URL of the file to download.
        output_path (str): Path to save the downloaded file.
        chunk_size (int): Chunk size for downloading the file.
        retries (int): Number of retries if the download fails.
        backoff (int): Time (in seconds) to wait between retries.
    """
    attempt = 0
    while attempt <= retries:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
            total_size = int(response.headers.get('content-length', 0))

            ensure_dir_exists(str(Path(output_path).parent))

            with open(output_path, "wb") as file, tqdm(
                desc=f"Downloading {Path(output_path).name}",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024
            ) as bar:
                for data in response.iter_content(chunk_size=chunk_size):
                    file.write(data)
                    bar.update(len(data))

            logger.info(f"File downloaded successfully: {output_path}")
            return  # Exit if the download is successful

        except RequestException as e:
            attempt += 1
            logger.warning(f"Failed to download {url} (Attempt {attempt}/{retries}). Error: {e}")
            if attempt <= retries:
                logger.info(f"Retrying in {backoff} seconds...")
                sleep(backoff)

    logger.error(f"Exceeded maximum retries. Failed to download {url}")

def parallel_download_files(urls: List[str], output_dir: str, max_workers: int = 4) -> None:
    """
    Download multiple files in parallel.

    Args:
        urls (List[str]): List of URLs to download.
        output_dir (str): Directory where the downloaded files will be saved.
        max_workers (int): Maximum number of worker threads to use for downloading.
    """
    ensure_dir_exists(output_dir)

    def _download_single_file(url: str):
        filename = Path(url).name
        output_path = Path(output_dir) / filename
        download_file(url, str(output_path))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_download_single_file, url) for url in urls]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error during download: {e}")

def list_files(directory: str, extension: Optional[str] = None) -> List[str]:
    """
    List all files in a directory, optionally filtering by file extension.

    Args:
        directory (str): Path to the directory.
        extension (str, optional): File extension to filter by (e.g., '.txt').

    Returns:
        List[str]: List of file paths.
    """
    path = Path(directory)
    if not path.is_dir():
        logger.error(f"Directory {directory} does not exist.")
        raise FileNotFoundError(f"Directory {directory} does not exist.")
    
    files = path.rglob(f'*{extension}' if extension else '*')
    return [str(file) for file in files if file.is_file()]

def move_file(src_path: str, dest_path: str) -> None:
    """
    Move a file from source path to destination path.

    Args:
        src_path (str): Path to the source file.
        dest_path (str): Path to the destination file.
    """
    try:
        src = Path(src_path)
        dest = Path(dest_path)
        ensure_dir_exists(str(dest.parent))
        shutil.move(str(src), str(dest))
        logger.info(f"Moved file from {src_path} to {dest_path}")
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"Error moving file from {src_path} to {dest_path}: {e}")
        raise

def copy_file(src_path: str, dest_path: str) -> None:
    """
    Copy a file from source path to destination path.

    Args:
        src_path (str): Path to the source file.
        dest_path (str): Path to the destination file.
    """
    try:
        src = Path(src_path)
        dest = Path(dest_path)
        ensure_dir_exists(str(dest.parent))
        shutil.copy(str(src), str(dest))
        logger.info(f"Copied file from {src_path} to {dest_path}")
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"Error copying file from {src_path} to {dest_path}: {e}")
        raise

def delete_file(file_path: str) -> None:
    """
    Delete the specified file.

    Args:
        file_path (str): Path to the file to be deleted.
    """
    try:
        file = Path(file_path)
        if file.exists():
            file.unlink()
            logger.info(f"Deleted file: {file_path}")
        else:
            logger.warning(f"File {file_path} does not exist.")
            raise FileNotFoundError(f"File {file_path} does not exist.")
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        raise

def get_file_size(file_path: str) -> int:
    """
    Get the size of a file in bytes.

    Args:
        file_path (str): Path to the file.

    Returns:
        int: Size of the file in bytes.
    """
    file = Path(file_path)
    if not file.exists():
        logger.error(f"File {file_path} does not exist.")
        raise FileNotFoundError(f"File {file_path} does not exist.")
    
    size = file.stat().st_size
    logger.info(f"File size of {file_path}: {size} bytes")
    return size

def is_file_empty(file_path: str) -> bool:
    """
    Check if a file is empty.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if the file is empty, False otherwise.
    """
    return get_file_size(file_path) == 0

def read_file(file_path: str, mode: str = "r") -> str:
    """
    Read the contents of a file.

    Args:
        file_path (str): Path to the file.
        mode (str): Mode to open the file (default is "r" for reading).

    Returns:
        str: Contents of the file.
    """
    try:
        file = Path(file_path)
        with file.open(mode) as f:
            content = f.read()
        logger.info(f"Read file: {file_path}")
        return content
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise

def write_file(file_path: str, content: str, mode: str = "w") -> None:
    """
    Write content to a file.

    Args:
        file_path (str): Path to the file.
        content (str): Content to write to the file.
        mode (str): Mode to open the file (default is "w" for writing).
    """
    try:
        file = Path(file_path)
        ensure_dir_exists(str(file.parent))
        with file.open(mode) as f:
            f.write(content)
        logger.info(f"Wrote to file: {file_path}")
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"Error writing to file {file_path}: {e}")
        raise

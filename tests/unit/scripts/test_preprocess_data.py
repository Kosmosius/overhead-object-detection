# tests/unit/scripts/test_preprocess_data.py

import os
import pytest
from unittest import mock
from PIL import Image
from scripts.preprocess_data import chip_image, adjust_gsd, preprocess_dataset
from src.data.augmentation import DataAugmentor
from src.utils.file_utils import ensure_dir_exists

# Unit test for chip_image function
def test_chip_image_normal():
    """Test normal chipping of an image into smaller sections."""
    img = Image.new('RGB', (1024, 1024))
    chip_size = (512, 512)
    overlap = 0.2

    chips, metadata = chip_image(img, chip_size, overlap)

    # Assert that 4 chips are created with correct dimensions
    assert len(chips) == 4
    assert all(chip.size == chip_size for chip in chips)
    assert len(metadata) == 4


def test_chip_image_edge_case():
    """Test chipping when image dimensions are not perfectly divisible by chip size."""
    img = Image.new('RGB', (1000, 1000))  # Image dimensions not divisible by 512
    chip_size = (512, 512)
    overlap = 0.2

    chips, metadata = chip_image(img, chip_size, overlap)

    # Assert that at least 4 chips are created, as expected
    assert len(chips) > 1
    assert len(metadata) == len(chips)


def test_chip_image_invalid_overlap():
    """Test chip_image with an invalid overlap value."""
    img = Image.new('RGB', (1024, 1024))
    chip_size = (512, 512)

    with pytest.raises(ValueError):
        chip_image(img, chip_size, overlap=-0.5)  # Invalid overlap


# Unit test for adjust_gsd function
def test_adjust_gsd_normal():
    """Test resizing the image based on GSD adjustment."""
    img = Image.new('RGB', (1024, 1024))
    original_gsd = 1.5
    target_gsd = 1.0

    resized_img = adjust_gsd(img, original_gsd, target_gsd)

    # Assert that image was resized correctly based on GSD ratio
    assert resized_img.size == (int(1024 * 1.5 / 1.0), int(1024 * 1.5 / 1.0))


def test_adjust_gsd_invalid_gsd():
    """Test GSD adjustment with invalid GSD values."""
    img = Image.new('RGB', (1024, 1024))

    with pytest.raises(ZeroDivisionError):
        adjust_gsd(img, 1.5, 0)  # Invalid target GSD


def test_adjust_gsd_no_change():
    """Test that adjust_gsd returns the same image when original and target GSD are equal."""
    img = Image.new('RGB', (1024, 1024))
    original_gsd = 1.0
    target_gsd = 1.0

    resized_img = adjust_gsd(img, original_gsd, target_gsd)

    # Assert that the image remains unchanged
    assert resized_img.size == img.size


# Unit test for preprocess_dataset function
@mock.patch('scripts.preprocess_data.adjust_gsd')
@mock.patch('scripts.preprocess_data.chip_image')
@mock.patch('scripts.preprocess_data.ensure_dir_exists')
@mock.patch('scripts.preprocess_data.Image.open')
@mock.patch('scripts.preprocess_data.DataAugmentor')
def test_preprocess_dataset(mock_augmentor, mock_image_open, mock_ensure_dir_exists, mock_chip_image, mock_adjust_gsd):
    """Test normal execution of preprocess_dataset, including chipping and GSD adjustment."""
    
    # Mock objects
    mock_img = mock.Mock(spec=Image.Image)
    mock_image_open.return_value = mock_img
    mock_chip_image.return_value = ([mock_img], [{}])
    
    # Run preprocess_dataset
    preprocess_dataset(
        data_dir='test_data/',
        output_dir='output_data/',
        chip_size=(512, 512),
        overlap=0.2,
        augment=False,
        original_gsd=1.5,
        target_gsd=1.0
    )
    
    # Assert directory creation, image opening, and chipping functions were called
    mock_ensure_dir_exists.assert_called_once_with('output_data/')
    mock_image_open.assert_called_once()
    mock_chip_image.assert_called_once_with(mock_img, (512, 512), 0.2)
    mock_adjust_gsd.assert_called_once()


# Edge case test: Missing images
@mock.patch('scripts.preprocess_data.os.listdir')
@mock.patch('scripts.preprocess_data.Image.open')
def test_preprocess_dataset_missing_images(mock_image_open, mock_listdir):
    """Test preprocess_dataset with missing or corrupted images."""
    # Simulate missing or corrupted image files
    mock_listdir.return_value = ['image1.png', 'corrupt_image.jpg']
    mock_image_open.side_effect = [Image.new('RGB', (1024, 1024)), IOError]  # IOError simulates a corrupted image

    # Run preprocess_dataset and ensure IOError is handled
    preprocess_dataset(
        data_dir='test_data/',
        output_dir='output_data/',
        chip_size=(512, 512),
        overlap=0.2,
        augment=False
    )
    assert mock_image_open.call_count == 2  # It tries to open both images


# Performance test: Processing large images
@mock.patch('scripts.preprocess_data.chip_image')
@mock.patch('scripts.preprocess_data.adjust_gsd')
def test_preprocess_large_image_performance(mock_adjust_gsd, mock_chip_image, benchmark):
    """Benchmark performance of large image processing."""
    
    img = Image.new('RGB', (5000, 5000))  # Simulate large image
    mock_chip_image.return_value = ([img], [{}])

    def run_preprocess():
        preprocess_dataset(
            data_dir='test_data/',
            output_dir='output_data/',
            chip_size=(512, 512),
            overlap=0.2,
            augment=False
        )
    
    benchmark(run_preprocess)  # Benchmark performance


# Edge case test: Invalid file formats
@mock.patch('scripts.preprocess_data.os.listdir')
@mock.patch('scripts.preprocess_data.Image.open')
def test_preprocess_invalid_image_formats(mock_image_open, mock_listdir):
    """Test preprocess_dataset handling unsupported image formats."""
    
    # Simulate unsupported image formats
    mock_listdir.return_value = ['unsupported_file.txt', 'image.jpg']
    mock_image_open.side_effect = [Image.new('RGB', (1024, 1024)), IOError]

    preprocess_dataset(
        data_dir='test_data/',
        output_dir='output_data/',
        chip_size=(512, 512),
        overlap=0.2,
        augment=False
    )

    # Only the valid image should be opened
    mock_image_open.assert_called_once_with('test_data/image.jpg')


# Mock data augmentation tests
@mock.patch('scripts.preprocess_data.DataAugmentor')
@mock.patch('scripts.preprocess_data.Image.open')
def test_preprocess_with_augmentation(mock_image_open, mock_data_augmentor):
    """Test data augmentation applied during preprocess_dataset."""
    mock_img = mock.Mock(spec=Image.Image)
    mock_image_open.return_value = mock_img
    mock_chip_image.return_value = ([mock_img], [{}])
    
    preprocess_dataset(
        data_dir='test_data/',
        output_dir='output_data/',
        chip_size=(512, 512),
        overlap=0.2,
        augment=True
    )
    
    # Assert that the augmentation was applied
    mock_data_augmentor.assert_called_once_with(apply_geometric=True, apply_photometric=True)
    mock_data_augmentor.return_value.apply_augmentation.assert_called()


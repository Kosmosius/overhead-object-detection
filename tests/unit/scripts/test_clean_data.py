# tests/unit/scripts/test_clean_data.py

import os
import json
import pytest
import shutil
from unittest import mock
from scripts.clean_data import ensure_dir_exists, clean_coco, clean_geojson, clean_overhead_dataset, clean_dota_dataset, clean_xview_dataset, clean_standard_overhead_dataset
from pycocotools.coco import COCO
from geojson import FeatureCollection

# Mock for tqdm so the progress bar doesn’t interfere with test output
@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    monkeypatch.setattr("tqdm.tqdm", lambda x, *args, **kwargs: x)

# Mock for os.makedirs to prevent actual file system changes
@mock.patch("os.makedirs")
def test_ensure_dir_exists(mock_makedirs):
    """Test directory creation function."""
    ensure_dir_exists("test_dir")
    mock_makedirs.assert_called_once_with("test_dir")

# Unit test for clean_coco
@mock.patch("pycocotools.coco.COCO")
@mock.patch("shutil.copy")
@mock.patch("os.path.exists")
def test_clean_coco(mock_exists, mock_copy, mock_coco):
    """Test cleaning COCO dataset."""
    mock_exists.return_value = True
    mock_coco.return_value.getImgIds.return_value = [1]
    mock_coco.return_value.loadImgs.return_value = [{"file_name": "image1.jpg"}]

    clean_coco("fake_annotations.json", "output_dir")

    mock_exists.assert_called_once_with("image1.jpg")
    mock_copy.assert_called_once_with("image1.jpg", "output_dir")

# Edge case test: COCO dataset with missing image files
@mock.patch("pycocotools.coco.COCO")
@mock.patch("os.path.exists")
def test_clean_coco_missing_image(mock_exists, mock_coco):
    """Test cleaning COCO dataset with missing image files."""
    mock_exists.return_value = False
    mock_coco.return_value.getImgIds.return_value = [1]
    mock_coco.return_value.loadImgs.return_value = [{"file_name": "missing_image.jpg"}]

    with pytest.warns(UserWarning, match="Missing image: missing_image.jpg"):
        clean_coco("fake_annotations.json", "output_dir")

# Unit test for clean_geojson
@mock.patch("builtins.open", new_callable=mock.mock_open, read_data='{"features": [{"geometry": {}, "properties": {}}]}')
@mock.patch("shutil.copy")
def test_clean_geojson(mock_copy, mock_open):
    """Test cleaning GeoJSON dataset."""
    clean_geojson("fake_geojson.geojson", "output_dir")
    mock_open.assert_called_once_with("fake_geojson.geojson", 'r')

# Edge case test: GeoJSON dataset with invalid features
@mock.patch("builtins.open", new_callable=mock.mock_open, read_data='{"features": [{}]}')
@mock.patch("shutil.copy")
def test_clean_geojson_invalid_feature(mock_copy, mock_open):
    """Test GeoJSON with invalid feature."""
    with pytest.warns(UserWarning, match="Invalid feature found: {}"):
        clean_geojson("fake_geojson.geojson", "output_dir")

# Unit test for clean_overhead_dataset (DOTA dataset case)
@mock.patch("os.path.exists")
@mock.patch("shutil.copy")
def test_clean_dota_dataset(mock_copy, mock_exists):
    """Test cleaning DOTA dataset."""
    mock_exists.return_value = True
    annotations = [{"image_name": "image1.jpg"}]
    with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(annotations))):
        clean_dota_dataset("annotations.json", "images", "output_dir")
    
    mock_copy.assert_called_once_with("images/image1.jpg", "output_dir")

# Edge case: Empty dataset
def test_clean_overhead_dataset_empty():
    """Test handling an empty overhead dataset."""
    with pytest.raises(ValueError):
        clean_overhead_dataset("", "", "", "")

# Performance test: Handling large datasets (mocked)
@mock.patch("pycocotools.coco.COCO")
@mock.patch("shutil.copy")
@mock.patch("os.path.exists")
def test_clean_coco_large_dataset(mock_exists, mock_copy, mock_coco):
    """Test performance for cleaning a large COCO dataset."""
    # Simulate a large number of images
    mock_exists.return_value = True
    mock_coco.return_value.getImgIds.return_value = list(range(1000))
    mock_coco.return_value.loadImgs.return_value = [{"file_name": f"image_{i}.jpg"} for i in range(1000)]

    clean_coco("fake_annotations.json", "output_dir")

    assert mock_copy.call_count == 1000

# Mock the file system for missing images in overhead datasets
@mock.patch("os.path.exists")
def test_clean_overhead_dataset_missing_image(mock_exists):
    """Test overhead dataset cleaning with missing image files."""
    mock_exists.return_value = False
    annotations = [{"image_name": "missing_image.jpg"}]
    with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(annotations))):
        with pytest.warns(UserWarning, match="Missing image: missing_image.jpg"):
            clean_dota_dataset("annotations.json", "images", "output_dir")

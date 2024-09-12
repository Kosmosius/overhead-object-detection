# tests/unit/src/training/test_peft_finetune.py

from src.training.peft_finetune import setup_peft_model, prepare_dataloader
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from peft import PeftConfig
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_peft_config():
    return MagicMock(spec=PeftConfig)

@pytest.fixture
def mock_feature_extractor():
    return MagicMock(spec=DetrFeatureExtractor)

def test_setup_peft_model(mock_peft_config):
    with patch("src.training.peft_finetune.DetrForObjectDetection.from_pretrained") as mock_from_pretrained:
        mock_from_pretrained.return_value = MagicMock(spec=DetrForObjectDetection)
        
        model = setup_peft_model("facebook/detr-resnet-50", num_classes=91, peft_config=mock_peft_config)
        
        assert model is not None, "Model should be initialized"
        mock_from_pretrained.assert_called_once()

def test_prepare_dataloader(mock_feature_extractor):
    with patch("src.training.peft_finetune.load_dataset") as mock_load_dataset:
        mock_load_dataset.return_value = MagicMock()  # Mock dataset
        dataloader = prepare_dataloader("", 2, mock_feature_extractor, mode="train")
        
        assert dataloader is not None, "Dataloader should be returned"
        mock_load_dataset.assert_called_once()

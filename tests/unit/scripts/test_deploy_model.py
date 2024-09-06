# tests/unit/scripts/test_deploy_model.py

import os
import pytest
from unittest import mock
from scripts.deploy_model import ensure_dir_exists, deploy_local_docker, deploy_cloud_service, deploy_to_aws, deploy_to_gcp, deploy_to_azure

# Mock for subprocess to prevent actual Docker commands from running
@mock.patch("subprocess.run")
def test_deploy_local_docker(mock_subprocess):
    """Test local model deployment using Docker."""
    # Mocking subprocess.run to prevent actual Docker build/run
    mock_subprocess.return_value = None

    deploy_local_docker("fake_model_path", port=8080)

    # Check that Docker build and run commands were called
    assert mock_subprocess.call_count == 2
    mock_subprocess.assert_any_call(["docker", "build", "-t", "model_deployment", "."], cwd="docker_deployment", check=True)
    mock_subprocess.assert_any_call(["docker", "run", "-p", "8080:5000", "model_deployment"], check=True)

# Edge case test: Invalid Docker deployment due to missing model
@mock.patch("subprocess.run")
def test_deploy_local_docker_missing_model(mock_subprocess):
    """Test Docker deployment when the model is missing."""
    mock_subprocess.side_effect = FileNotFoundError("Model not found.")

    with pytest.raises(FileNotFoundError, match="Model not found."):
        deploy_local_docker("missing_model_path", port=8080)

# Unit test for ensure_dir_exists function
@mock.patch("os.makedirs")
@mock.patch("os.path.exists", return_value=False)
def test_ensure_dir_exists(mock_exists, mock_makedirs):
    """Test that the directory is created if it doesn't exist."""
    ensure_dir_exists("test_directory")
    mock_makedirs.assert_called_once_with("test_directory")

# Edge case: Directory already exists
@mock.patch("os.makedirs")
@mock.patch("os.path.exists", return_value=True)
def test_ensure_dir_exists_directory_exists(mock_exists, mock_makedirs):
    """Test that the function does nothing if the directory already exists."""
    ensure_dir_exists("test_directory")
    mock_makedirs.assert_not_called()

# Mock for cloud deployments
@mock.patch("scripts.deploy_model.deploy_to_aws")
@mock.patch("scripts.deploy_model.deploy_to_gcp")
@mock.patch("scripts.deploy_model.deploy_to_azure")
def test_deploy_cloud_service_aws(mock_azure, mock_gcp, mock_aws):
    """Test cloud deployment via AWS."""
    deploy_cloud_service("fake_model_path", cloud_provider="aws", service_name="aws_service")

    mock_aws.assert_called_once_with("fake_model_path", "aws_service")
    mock_gcp.assert_not_called()
    mock_azure.assert_not_called()

def test_deploy_cloud_service_invalid_provider():
    """Test cloud deployment with an unsupported cloud provider."""
    with pytest.raises(ValueError, match="Unsupported cloud provider"):
        deploy_cloud_service("fake_model_path", cloud_provider="unsupported", service_name="unsupported_service")

# Functional tests for cloud service deployment mocks
@mock.patch("subprocess.run")
@mock.patch("scripts.deploy_model.deploy_to_aws")
def test_deploy_to_aws(mock_deploy_to_aws, mock_subprocess):
    """Test AWS deployment (mocked)."""
    deploy_to_aws("fake_model_path", "aws_service")
    mock_deploy_to_aws.assert_called_once_with("fake_model_path", "aws_service")

@mock.patch("subprocess.run")
@mock.patch("scripts.deploy_model.deploy_to_gcp")
def test_deploy_to_gcp(mock_deploy_to_gcp, mock_subprocess):
    """Test GCP deployment (mocked)."""
    deploy_to_gcp("fake_model_path", "gcp_service")
    mock_deploy_to_gcp.assert_called_once_with("fake_model_path", "gcp_service")

@mock.patch("subprocess.run")
@mock.patch("scripts.deploy_model.deploy_to_azure")
def test_deploy_to_azure(mock_deploy_to_azure, mock_subprocess):
    """Test Azure deployment (mocked)."""
    deploy_to_azure("fake_model_path", "azure_service")
    mock_deploy_to_azure.assert_called_once_with("fake_model_path", "azure_service")

# Integration test: Full deployment workflow
@mock.patch("scripts.deploy_model.deploy_local_docker")
@mock.patch("scripts.deploy_model.deploy_cloud_service")
def test_full_deployment_workflow(mock_deploy_cloud, mock_deploy_local):
    """Test the full deployment workflow including both local and cloud deployments."""
    # Test local deployment
    deploy_local_docker("model_path", port=8080)
    mock_deploy_local.assert_called_once_with("model_path", 8080)

    # Test cloud deployment (AWS example)
    deploy_cloud_service("model_path", cloud_provider="aws", service_name="aws_service")
    mock_deploy_cloud.assert_called_once_with("model_path", "aws", "aws_service")

# Edge case: Cloud deployment fails
@mock.patch("scripts.deploy_model.deploy_to_aws", side_effect=Exception("AWS deployment failed"))
def test_deploy_cloud_service_aws_failure(mock_aws):
    """Test cloud deployment failure (AWS)."""
    with pytest.raises(Exception, match="AWS deployment failed"):
        deploy_cloud_service("fake_model_path", cloud_provider="aws", service_name="aws_service")

# Performance test (mocked): Handle a large number of requests
@mock.patch("subprocess.run")
def test_performance_large_docker_deploy(mock_subprocess):
    """Test handling a large number of Docker deployments in sequence (performance test)."""
    for _ in range(100):
        deploy_local_docker("fake_model_path", port=8080)

    assert mock_subprocess.call_count == 200  # Each deploy involves two subprocess calls

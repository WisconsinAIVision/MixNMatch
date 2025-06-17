import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import json


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config():
    """Provide a mock configuration object."""
    config = Mock()
    config.debug = False
    config.output_dir = "/tmp/test_output"
    config.batch_size = 32
    config.learning_rate = 0.001
    return config


@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    return {
        "images": ["image1.png", "image2.png", "image3.png"],
        "labels": [0, 1, 2],
        "metadata": {
            "dataset": "test",
            "version": "1.0"
        }
    }


@pytest.fixture
def mock_model():
    """Provide a mock model for testing."""
    model = Mock()
    model.predict = Mock(return_value=[0.1, 0.8, 0.1])
    model.train = Mock()
    model.evaluate = Mock(return_value={"accuracy": 0.95, "loss": 0.05})
    return model


@pytest.fixture
def test_image_path(temp_dir):
    """Create a test image file."""
    image_path = temp_dir / "test_image.png"
    image_path.write_bytes(b"fake image data")
    return image_path


@pytest.fixture
def json_file(temp_dir):
    """Create a temporary JSON file."""
    json_path = temp_dir / "test_data.json"
    
    def _create_json(data):
        with open(json_path, 'w') as f:
            json.dump(data, f)
        return json_path
    
    return _create_json


@pytest.fixture
def capture_logs():
    """Capture log messages during tests."""
    logs = []
    
    def log_capture(message, level="INFO"):
        logs.append({"message": message, "level": level})
    
    return log_capture, logs


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch):
    """Reset environment variables for each test."""
    monkeypatch.setenv("PYTHONPATH", str(Path.cwd()))
    monkeypatch.setenv("TEST_ENV", "true")


@pytest.fixture
def mock_file_operations():
    """Mock common file operations."""
    with patch('builtins.open', create=True) as mock_open:
        with patch('os.path.exists') as mock_exists:
            with patch('os.makedirs') as mock_makedirs:
                yield {
                    'open': mock_open,
                    'exists': mock_exists,
                    'makedirs': mock_makedirs
                }


def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "requires_network: mark test as requiring network access"
    )
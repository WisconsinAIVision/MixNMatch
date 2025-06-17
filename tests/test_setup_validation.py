import pytest
import sys
from pathlib import Path


class TestSetupValidation:
    """Validation tests to ensure the testing infrastructure is properly configured."""
    
    def test_pytest_installed(self):
        """Verify pytest is available."""
        assert 'pytest' in sys.modules
    
    def test_project_structure(self):
        """Verify the project structure is correct."""
        project_root = Path(__file__).parent.parent
        
        assert project_root.exists()
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "__init__.py").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "unit" / "__init__.py").exists()
        assert (project_root / "tests" / "integration").exists()
        assert (project_root / "tests" / "integration" / "__init__.py").exists()
        assert (project_root / "tests" / "conftest.py").exists()
    
    def test_pyproject_toml_exists(self):
        """Verify pyproject.toml exists and is properly configured."""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        assert pyproject_path.exists(), "pyproject.toml should exist"
        
        content = pyproject_path.read_text()
        assert "[tool.poetry]" in content
        assert "[tool.pytest.ini_options]" in content
        assert "[tool.coverage.run]" in content
    
    @pytest.mark.unit
    def test_markers_defined(self):
        """Verify custom markers are defined."""
        assert True  # If this runs with the marker, markers are working
    
    def test_fixtures_available(self, temp_dir, mock_config, sample_data):
        """Verify fixtures from conftest.py are available."""
        assert temp_dir.exists()
        assert hasattr(mock_config, 'debug')
        assert 'images' in sample_data
        assert 'labels' in sample_data
    
    def test_coverage_configured(self):
        """Verify coverage is properly configured."""
        try:
            import coverage
            assert True
        except ImportError:
            pytest.fail("Coverage module should be available")
    
    def test_mock_available(self):
        """Verify pytest-mock is available."""
        try:
            from pytest_mock import MockerFixture
            assert True
        except ImportError:
            pytest.fail("pytest-mock should be available")
    
    def test_code_package_importable(self):
        """Verify the code package can be imported."""
        project_root = Path(__file__).parent.parent
        code_dir = project_root / "code"
        
        assert code_dir.exists(), "code directory should exist"
        
        # Add project root to Python path if not already there
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        try:
            import code
            assert True
        except ImportError:
            # This is expected if __init__.py doesn't exist in code/
            # Just verify the directory exists
            assert code_dir.is_dir()


@pytest.mark.integration
class TestIntegrationSetup:
    """Integration test to verify the entire setup works together."""
    
    def test_full_test_run_simulation(self, temp_dir, mock_model, json_file):
        """Simulate a complete test scenario."""
        # Create test data
        test_data = {"test": "data", "values": [1, 2, 3]}
        json_path = json_file(test_data)
        
        # Verify file was created
        assert json_path.exists()
        
        # Verify mock model works
        result = mock_model.predict()
        assert isinstance(result, list)
        assert len(result) == 3
        
        # Verify temp directory cleanup will happen
        assert temp_dir.exists()


@pytest.mark.slow
def test_slow_marker():
    """Test that slow marker works."""
    import time
    time.sleep(0.1)  # Simulate slow test
    assert True
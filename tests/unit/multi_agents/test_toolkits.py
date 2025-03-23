import pytest
from unittest.mock import MagicMock, patch

from oumi.multi_agents.toolkits import create_toolkit, get_available_toolkits


@pytest.fixture
def mock_toolkits():
    """Setup mock toolkits for testing."""
    with patch("oumi.multi_agents.toolkits.AVAILABLE_TOOLKITS", {
        "MathToolkit": MagicMock(),
        "WeatherToolkit": MagicMock(),
        "DalleToolkit": MagicMock(),
    }):
        yield


def test_get_available_toolkits(mock_toolkits):
    """Test getting the list of available toolkits."""
    toolkits = get_available_toolkits()
    assert isinstance(toolkits, list)
    assert "MathToolkit" in toolkits
    assert "WeatherToolkit" in toolkits
    assert "DalleToolkit" in toolkits
    assert len(toolkits) == 3


def test_create_toolkit_success(mock_toolkits):
    """Test creating a toolkit successfully."""
    # Setup mock toolkit class
    mock_toolkit_class = MagicMock()
    mock_toolkit_instance = MagicMock()
    mock_toolkit_class.return_value = mock_toolkit_instance
    
    # Replace the MathToolkit class with our mock
    with patch("oumi.multi_agents.toolkits.AVAILABLE_TOOLKITS", {
        "MathToolkit": mock_toolkit_class,
    }):
        # Create the toolkit
        toolkit = create_toolkit("MathToolkit")
        
        # Verify the toolkit was created correctly
        mock_toolkit_class.assert_called_once_with()
        assert toolkit == mock_toolkit_instance


def test_create_toolkit_with_api_key(mock_toolkits):
    """Test creating a toolkit with an API key."""
    # Setup mock toolkit class
    mock_toolkit_class = MagicMock()
    mock_toolkit_instance = MagicMock()
    mock_toolkit_class.return_value = mock_toolkit_instance
    
    # Replace the WeatherToolkit class with our mock
    with patch("oumi.multi_agents.toolkits.AVAILABLE_TOOLKITS", {
        "WeatherToolkit": mock_toolkit_class,
    }):
        # Create the toolkit with an API key
        api_key = "test_api_key_123"
        toolkit = create_toolkit("WeatherToolkit", api_key=api_key)
        
        # Verify the toolkit was created with the API key
        mock_toolkit_class.assert_called_once_with(api_key=api_key)
        assert toolkit == mock_toolkit_instance


def test_create_toolkit_with_nonexistent_toolkit():
    """Test creating a toolkit that doesn't exist."""
    with pytest.raises(ValueError, match="Toolkit 'NonexistentToolkit' not found"):
        create_toolkit("NonexistentToolkit")


def test_create_openai_toolkit_without_api_key(mock_toolkits):
    """Test creating an OpenAI toolkit without providing an API key."""
    # Setup mock toolkit class that requires an API key
    mock_toolkit_class = MagicMock()
    
    # Replace the DalleToolkit class with our mock
    with patch("oumi.multi_agents.toolkits.AVAILABLE_TOOLKITS", {
        "DalleToolkit": mock_toolkit_class,
    }):
        # Attempt to create the toolkit without an API key
        with pytest.raises(ValueError, match="API key is required for DalleToolkit"):
            # Mock the toolkit's behavior of raising ValueError when no API key is provided
            mock_toolkit_class.side_effect = ValueError("API key is required for DalleToolkit")
            create_toolkit("DalleToolkit")


@patch("oumi.multi_agents.toolkits.MathToolkit")
def test_math_toolkit_functionality(mock_math_toolkit):
    """Test the functionality of the MathToolkit."""
    # Setup the mock toolkit
    mock_toolkit_instance = MagicMock()
    mock_math_toolkit.return_value = mock_toolkit_instance
    
    # Define expected tool functions
    mock_toolkit_instance.tools = [
        {"name": "calculate", "description": "Performs calculations"},
        {"name": "solve_equation", "description": "Solves equations"},
    ]
    
    # Create the toolkit
    with patch("oumi.multi_agents.toolkits.AVAILABLE_TOOLKITS", {
        "MathToolkit": mock_math_toolkit,
    }):
        toolkit = create_toolkit("MathToolkit")
        
        # Verify the toolkit has the expected tools
        assert toolkit.tools == mock_toolkit_instance.tools
        assert len(toolkit.tools) == 2
        
        # Test a toolkit function
        input_data = {"expression": "2 + 2"}
        expected_result = 4
        
        # Configure the mock to return the expected result
        mock_toolkit_instance.calculate.return_value = expected_result
        
        # Call the function and verify the result
        result = toolkit.calculate(input_data)
        assert result == expected_result
        mock_toolkit_instance.calculate.assert_called_once_with(input_data) 
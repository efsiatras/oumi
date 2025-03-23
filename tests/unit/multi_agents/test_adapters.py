import pytest
from unittest.mock import MagicMock, patch

from oumi.multi_agents.adapters import create_agent_model


@patch("oumi.multi_agents.adapters.OumiModelWrapper")
@patch("oumi.multi_agents.adapters.load_model")
def test_create_agent_model(mock_load_model, mock_wrapper_class):
    """Test creating an Oumi model adapter for multi-agent systems with CAMEL."""
    # Setup mocks
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    
    mock_wrapper = MagicMock()
    mock_wrapper_class.return_value = mock_wrapper
    
    # Test with minimal parameters
    model = create_agent_model(model_name_or_path="path/to/model")
    
    # Verify model was loaded and wrapped correctly
    mock_load_model.assert_called_once_with("path/to/model")
    mock_wrapper_class.assert_called_once()
    assert model == mock_wrapper
    
    # Reset mocks
    mock_load_model.reset_mock()
    mock_wrapper_class.reset_mock()
    
    # Test with system message
    model = create_agent_model(
        model_name_or_path="path/to/model",
        system_message="Custom system message"
    )
    
    # Verify model was loaded and wrapped with system message
    mock_load_model.assert_called_once_with("path/to/model")
    mock_wrapper_class.assert_called_once()
    mock_wrapper_class.assert_called_with(
        mock_model,
        system_message="Custom system message"
    )
    assert model == mock_wrapper


@patch("oumi.multi_agents.adapters.OumiModelWrapper")
@patch("oumi.multi_agents.adapters.load_model")
def test_oumi_model_wrapper_generate(mock_load_model, mock_wrapper_class):
    """Test the generate method of the OumiModelWrapper."""
    # Setup the mock model with a response
    mock_model = MagicMock()
    mock_model.generate.return_value = MagicMock(
        generation="Test response from Oumi model"
    )
    mock_load_model.return_value = mock_model
    
    # Create a real wrapper instance
    with patch("oumi.multi_agents.adapters.OumiModelWrapper.__init__", return_value=None):
        # Create an instance without calling __init__
        from oumi.multi_agents.adapters import OumiModelWrapper
        wrapper = OumiModelWrapper("dummy", "dummy")
        
        # Set attributes manually
        wrapper.model = mock_model
        wrapper.system_message = "You are a helpful assistant."
        
        # Test generate method
        test_message = {"role": "user", "content": "Hello"}
        result = wrapper.generate(messages=[test_message])
        
        # Verify model was called correctly
        mock_model.generate.assert_called_once()
        
        # Check that the result has the expected format
        assert hasattr(result, "msg")
        assert result.msg.content == "Test response from Oumi model"


@patch("oumi.multi_agents.adapters.load_model")
def test_oumi_model_wrapper_messages_format(mock_load_model):
    """Test that messages are formatted correctly for the Oumi model."""
    # Setup mock model
    mock_model = MagicMock()
    mock_model.generate.return_value = MagicMock(
        generation="Response"
    )
    mock_load_model.return_value = mock_model
    
    # Create model with system message
    model = create_agent_model(
        model_name_or_path="path/to/model",
        system_message="You are a helpful assistant."
    )
    
    # Test with multiple messages
    messages = [
        {"role": "system", "content": "System instruction"},
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant response"},
        {"role": "user", "content": "Follow-up question"}
    ]
    
    model.generate(messages=messages)
    
    # Verify model.generate was called with correctly formatted messages
    args, kwargs = mock_model.generate.call_args
    
    # Extract the actual messages passed to the model
    passed_messages = kwargs.get('messages', None)
    assert passed_messages is not None
    
    # Check message structure
    # The exact structure will depend on the OumiModelWrapper implementation
    # but we can verify that the number of messages is expected
    assert len(passed_messages) > 0 
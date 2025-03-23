import pytest
from unittest.mock import MagicMock, patch

from oumi.multi_agents import ChatAgent
from oumi.multi_agents.adapters import create_agent_model


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    mock = MagicMock()
    mock.generate.return_value = MagicMock(msg=MagicMock(content="This is a test response"))
    return mock


def test_chat_agent_initialization():
    """Test the initialization of a ChatAgent."""
    # Test with default parameters
    agent = ChatAgent("You are a helpful assistant.")
    assert agent.system_message == "You are a helpful assistant."
    assert agent.model is None
    
    # Test with custom model
    mock_model = MagicMock()
    agent = ChatAgent("Custom system message", model=mock_model)
    assert agent.system_message == "Custom system message"
    assert agent.model == mock_model


def test_chat_agent_reset():
    """Test resetting the chat agent."""
    agent = ChatAgent("You are a helpful assistant.")
    agent.reset()
    # Verify the chat history is empty after reset
    assert len(agent.messages) == 0


def test_chat_agent_step(mock_model):
    """Test the step method of the chat agent."""
    agent = ChatAgent("You are a helpful assistant.", model=mock_model)
    agent.reset()
    
    # Test making a step
    response = agent.step("Hello, agent!")
    
    # Verify the model was called with the correct messages
    mock_model.generate.assert_called_once()
    assert response.msg.content == "This is a test response"
    
    # Verify messages were added to the history
    assert len(agent.messages) == 2
    assert agent.messages[0]["content"] == "Hello, agent!"
    assert agent.messages[1]["content"] == "This is a test response"


@patch("oumi.multi_agents.adapters.create_agent_model")
def test_chat_agent_with_oumi_model(mock_create_model):
    """Test ChatAgent with an Oumi model."""
    # Setup the mock
    mock_model = MagicMock()
    mock_model.generate.return_value = MagicMock(msg=MagicMock(content="Oumi model response"))
    mock_create_model.return_value = mock_model
    
    # Create the agent with an Oumi model path
    model = create_agent_model(
        model_name_or_path="path/to/oumi/model",
        system_message="You are a helpful assistant."
    )
    agent = ChatAgent("You are a helpful assistant.", model=model)
    agent.reset()
    
    # Test the agent
    response = agent.step("Hello from Oumi!")
    assert response.msg.content == "Oumi model response"


def test_chat_agent_with_toolkit(mock_model):
    """Test adding a toolkit to a chat agent."""
    agent = ChatAgent("You are a helpful assistant.", model=mock_model)
    
    # Create a mock toolkit
    mock_toolkit = MagicMock()
    mock_toolkit.name = "TestToolkit"
    
    # Add the toolkit to the agent
    agent.add_toolkit(mock_toolkit)
    
    # Verify the toolkit was added
    assert mock_toolkit in agent.toolkits
    
    # Test step with toolkit
    agent.reset()
    response = agent.step("Use the toolkit")
    
    # Verify the model was called with the toolkit
    mock_model.generate.assert_called_once()
    assert response.msg.content == "This is a test response" 
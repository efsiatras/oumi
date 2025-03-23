import pytest
from unittest.mock import MagicMock, patch

from oumi.multi_agents import RolePlayingSession
from oumi.multi_agents.adapters import create_agent_model


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    mock = MagicMock()
    mock.generate.return_value = MagicMock(
        msg=MagicMock(content="This is a test response"),
        terminated=False,
        info={},
    )
    return mock


def test_role_playing_session_initialization():
    """Test the initialization of a RolePlayingSession."""
    # Test with default parameters
    session = RolePlayingSession(
        task_prompt="Design a machine learning model",
        assistant_role_name="Machine Learning Engineer",
        user_role_name="Product Manager",
    )
    assert session.task_prompt == "Design a machine learning model"
    assert session.assistant_role_name == "Machine Learning Engineer"
    assert session.user_role_name == "Product Manager"
    assert session.model is None
    assert session.chat_turn_limit == 10  # Default value
    
    # Test with custom parameters
    mock_model = MagicMock()
    session = RolePlayingSession(
        task_prompt="Custom task",
        assistant_role_name="Developer",
        user_role_name="Client",
        model=mock_model,
        chat_turn_limit=5,
    )
    assert session.task_prompt == "Custom task"
    assert session.assistant_role_name == "Developer"
    assert session.user_role_name == "Client"
    assert session.model == mock_model
    assert session.chat_turn_limit == 5


@patch("oumi.multi_agents.ChatAgent")
def test_role_playing_session_execute_conversation(mock_agent_class, mock_model):
    """Test executing a conversation in a role-playing session."""
    # Setup mock agents
    mock_assistant = MagicMock()
    mock_user = MagicMock()
    
    # Setup message responses
    assistant_response = MagicMock(
        msg=MagicMock(content="I'll help design that"),
        terminated=False,
        info={},
    )
    user_response = MagicMock(
        msg=MagicMock(content="Thank you for your help"),
        terminated=False,
        info={},
    )
    
    # Configure mock agents to return responses
    mock_assistant.step.return_value = assistant_response
    mock_user.step.return_value = user_response
    
    # Configure mock agent class to return our mock agents
    mock_agent_class.side_effect = [mock_assistant, mock_user]
    
    # Create the session
    session = RolePlayingSession(
        task_prompt="Design a machine learning model",
        assistant_role_name="Machine Learning Engineer",
        user_role_name="Product Manager",
        model=mock_model,
        chat_turn_limit=2,  # Limit to 2 turns for testing
    )
    
    # Execute the conversation
    conversation = session.execute_full_conversation()
    
    # Verify agents were created correctly
    assert mock_agent_class.call_count == 2
    
    # Verify conversation structure
    assert len(conversation) > 0
    # Check for system messages and task prompt
    system_messages = [msg for msg in conversation if "system" in msg["role"]]
    assert len(system_messages) >= 2  # At least assistant and user system messages
    
    # Check for the task message
    task_messages = [msg for msg in conversation if "task" in msg["role"]]
    assert len(task_messages) == 1
    assert task_messages[0]["content"] == "Design a machine learning model"
    
    # Verify agent interactions
    assert mock_assistant.step.call_count > 0
    assert mock_user.step.call_count > 0


def test_role_playing_session_step(mock_model):
    """Test the step method of a role-playing session."""
    # Create mock agents and message
    mock_assistant = MagicMock()
    mock_user = MagicMock()
    
    # Configure responses
    assistant_response = MagicMock(
        msg=MagicMock(content="Assistant response"),
        terminated=False,
        info={},
    )
    user_response = MagicMock(
        msg=MagicMock(content="User response"),
        terminated=False,
        info={},
    )
    
    mock_assistant.step.return_value = assistant_response
    mock_user.step.return_value = user_response
    
    # Create the session with mock agents
    session = RolePlayingSession(
        task_prompt="Test task",
        assistant_role_name="Assistant",
        user_role_name="User",
        model=mock_model,
    )
    
    # Replace the agents with our mocks
    session.assistant_agent = mock_assistant
    session.user_agent = mock_user
    
    # Create a mock input message
    input_msg = MagicMock(content="Input message")
    
    # Test the step method
    assistant_result, user_result = session.step(input_msg)
    
    # Verify the results
    assert assistant_result == assistant_response
    assert user_result == user_response
    
    # Verify the agents were called
    mock_assistant.step.assert_called_once_with(input_msg)
    mock_user.step.assert_called_once_with(assistant_response.msg)


@patch("oumi.multi_agents.adapters.create_agent_model")
def test_role_playing_with_oumi_model(mock_create_model):
    """Test RolePlayingSession with an Oumi model."""
    # Setup the mock
    mock_model = MagicMock()
    mock_create_model.return_value = mock_model
    
    # Create the session with an Oumi model path
    model = create_agent_model(
        model_name_or_path="path/to/oumi/model"
    )
    
    session = RolePlayingSession(
        task_prompt="Design with Oumi",
        assistant_role_name="Engineer",
        user_role_name="Client",
        model=model,
    )
    
    # Verify the model was created and used
    mock_create_model.assert_called_once_with(model_name_or_path="path/to/oumi/model")
    assert session.model == mock_model 
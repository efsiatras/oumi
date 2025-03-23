import pytest
from unittest.mock import MagicMock, patch

from oumi.multi_agents import RolePlaying
from oumi.multi_agents.api import (
    TaskPrompt, 
    Task, 
    PromptTemplateGenerator, 
    TaskType,
    DEFAULT_ROLE_PROMPT_TEMPLATE,
    GENERAL_ASSISTANT_ROLES,
    GENERAL_USER_ROLES
)


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


def test_prompt_template_generator():
    """Test the PromptTemplateGenerator."""
    generator = PromptTemplateGenerator()
    
    # Test getting task specify prompt for AI Society
    ai_society_prompt = generator.get_task_specify_prompt(TaskType.AI_SOCIETY)
    assert isinstance(ai_society_prompt, object)
    assert hasattr(ai_society_prompt, "get_system_prompt")
    assert hasattr(ai_society_prompt, "format")
    
    # Test getting task specify prompt for CODE
    code_prompt = generator.get_task_specify_prompt(TaskType.CODE)
    assert isinstance(code_prompt, object)
    
    # Test getting task specify prompt for MISALIGNMENT
    misalignment_prompt = generator.get_task_specify_prompt(TaskType.MISALIGNMENT)
    assert isinstance(misalignment_prompt, object)


def test_task_creation():
    """Test creating a Task object."""
    # Create a task prompt
    task_prompt = TaskPrompt(
        system_prompt="You are a task creator",
        content="Create a task for a Python Programmer and Product Manager"
    )
    
    # Create a task
    task = Task(task_prompt=task_prompt)
    
    # Verify task properties
    assert task.task_prompt == task_prompt
    assert task.specified_task_prompt is None  # Not yet specified


@patch("oumi.multi_agents.ChatAgent")
def test_role_playing_initialization(mock_agent_class, mock_model):
    """Test initializing the RolePlaying class."""
    # Setup mock agents
    mock_assistant = MagicMock()
    mock_user = MagicMock()
    mock_task_specify = MagicMock()
    
    # Configure mock agent class to return our mocks
    mock_agent_class.side_effect = [mock_assistant, mock_user, mock_task_specify]
    
    # Create the role-playing session
    role_play = RolePlaying(
        assistant_role_name="Python Programmer",
        user_role_name="Product Manager",
        task_prompt="Create a data visualization app",
        assistant_agent_kwargs={"model": mock_model},
        user_agent_kwargs={"model": mock_model},
        with_task_specify=True,
        task_specify_agent_kwargs={"model": mock_model},
    )
    
    # Verify the role-playing session was initialized correctly
    assert role_play.assistant_role_name == "Python Programmer"
    assert role_play.user_role_name == "Product Manager"
    assert role_play.task_prompt == "Create a data visualization app"
    assert mock_agent_class.call_count == 3  # Two for the agents, one for task specify
    
    # Verify system messages were created
    assert "Python Programmer" in role_play.assistant_sys_msg
    assert "Product Manager" in role_play.user_sys_msg


def test_default_role_prompt_template():
    """Test the DEFAULT_ROLE_PROMPT_TEMPLATE."""
    # Format the template with roles
    formatted = DEFAULT_ROLE_PROMPT_TEMPLATE.format(
        assistant_role="Engineer",
        user_role="Client"
    )
    
    # Verify the formatting worked
    assert "Engineer" in formatted
    assert "Client" in formatted


def test_general_roles():
    """Test that GENERAL_ASSISTANT_ROLES and GENERAL_USER_ROLES are available."""
    # Verify that the role lists are available and non-empty
    assert isinstance(GENERAL_ASSISTANT_ROLES, list)
    assert len(GENERAL_ASSISTANT_ROLES) > 0
    
    assert isinstance(GENERAL_USER_ROLES, list)
    assert len(GENERAL_USER_ROLES) > 0
    
    # Verify that common roles are included
    assert "Python Programmer" in GENERAL_ASSISTANT_ROLES
    assert "Product Manager" in GENERAL_USER_ROLES


@patch("oumi.multi_agents.RolePlaying.step")
def test_role_playing_init_chat(mock_step):
    """Test the init_chat method of RolePlaying."""
    # Setup mocks
    mock_step.return_value = (
        MagicMock(msg=MagicMock(content="Assistant response")),
        MagicMock(msg=MagicMock(content="User response"))
    )
    
    # Create the role-playing session
    role_play = RolePlaying(
        assistant_role_name="Designer",
        user_role_name="Client",
        task_prompt="Design a logo",
    )
    
    # Initialize the chat
    initial_msg = role_play.init_chat()
    
    # Verify the initial message
    assert initial_msg is not None
    
    # Verify step was called
    mock_step.assert_called_once()


@patch("oumi.multi_agents.ChatAgent")
def test_role_playing_with_custom_task(mock_agent_class):
    """Test RolePlaying with a custom task."""
    # Setup mock agents
    mock_assistant = MagicMock()
    mock_user = MagicMock()
    
    # Configure mock agent class to return our mocks
    mock_agent_class.side_effect = [mock_assistant, mock_user]
    
    # Create a custom task
    task_prompt = TaskPrompt(
        system_prompt="You are a task creator",
        content="Create a custom task for an AI researcher and a philosopher"
    )
    task = Task(task_prompt=task_prompt)
    
    # Create the role-playing session with the custom task
    role_play = RolePlaying(
        assistant_role_name="AI Researcher",
        user_role_name="Philosopher",
        task=task,
        with_task_specify=False,  # No need for task specification since we provide a task
    )
    
    # Verify the task was set correctly
    assert role_play.task == task
    assert not role_play.with_task_specify 
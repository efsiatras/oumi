"""
Shared fixtures for testing the multi_agents module.
"""

import pytest
from unittest.mock import MagicMock


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


@pytest.fixture
def mock_chat_agent():
    """Create a mock ChatAgent for testing."""
    mock = MagicMock()
    mock.step.return_value = MagicMock(
        msg=MagicMock(content="Agent response"),
        terminated=False,
        info={},
    )
    mock.reset.return_value = None
    mock.messages = []
    return mock


@pytest.fixture
def mock_interpreter():
    """Create a mock code interpreter."""
    mock = MagicMock()
    mock.run.return_value = {
        "output": "Code execution result",
        "error": None,
        "status": "success"
    }
    return mock


@pytest.fixture
def mock_toolkit():
    """Create a mock toolkit for testing."""
    mock = MagicMock()
    mock.name = "MockToolkit"
    mock.tools = [
        {"name": "test_tool", "description": "A test tool"},
    ]
    return mock


@pytest.fixture
def mock_task():
    """Create a mock Task for testing."""
    mock = MagicMock()
    mock.task_prompt = MagicMock(
        system_prompt="You are a task creator",
        content="Create a test task"
    )
    mock.specified_task_prompt = None
    return mock 
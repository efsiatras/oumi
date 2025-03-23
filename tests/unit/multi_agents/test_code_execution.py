import pytest
from unittest.mock import MagicMock, patch

from oumi.multi_agents.api import CodeExecutionToolkit, InternalPythonInterpreter


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


def test_internal_python_interpreter():
    """Test the InternalPythonInterpreter."""
    interpreter = InternalPythonInterpreter()
    
    # Test basic code execution
    result = interpreter.run("print('Hello, world!')")
    assert "output" in result
    assert "Hello, world!" in result["output"]
    assert result["error"] is None
    assert result["status"] == "success"
    
    # Test code with error
    result = interpreter.run("print(undefined_variable)")
    assert "error" in result
    assert result["status"] == "error"
    assert "NameError" in result["error"]


def test_code_execution_toolkit_initialization():
    """Test initializing the CodeExecutionToolkit."""
    # Test with default interpreter
    toolkit = CodeExecutionToolkit()
    assert toolkit.name == "CodeExecutionToolkit"
    assert toolkit.interpreter is not None
    
    # Test with custom interpreter
    mock_interpreter = MagicMock()
    toolkit = CodeExecutionToolkit(interpreter=mock_interpreter)
    assert toolkit.interpreter == mock_interpreter


def test_code_execution_toolkit_tools():
    """Test that the CodeExecutionToolkit has the expected tools."""
    toolkit = CodeExecutionToolkit()
    
    # Verify the toolkit has tools
    assert hasattr(toolkit, "tools")
    assert len(toolkit.tools) > 0
    
    # Verify the toolkit has an execute_python tool
    tool_names = [tool["name"] for tool in toolkit.tools]
    assert "execute_python" in tool_names
    
    # Find the execute_python tool
    execute_python_tool = next(tool for tool in toolkit.tools if tool["name"] == "execute_python")
    assert "description" in execute_python_tool
    assert "parameters" in execute_python_tool


def test_execute_python_function(mock_interpreter):
    """Test the execute_python function of the CodeExecutionToolkit."""
    # Create toolkit with mock interpreter
    toolkit = CodeExecutionToolkit(interpreter=mock_interpreter)
    
    # Test executing Python code
    code = "def add(a, b): return a + b\nresult = add(2, 3)\nprint(result)"
    result = toolkit.execute_python({"code": code})
    
    # Verify the interpreter was called with the correct code
    mock_interpreter.run.assert_called_once_with(code)
    
    # Verify the result has the expected structure
    assert result["output"] == "Code execution result"
    assert result["error"] is None
    assert result["status"] == "success"


def test_execute_python_with_error():
    """Test executing Python code that contains an error."""
    # Create toolkit with a real interpreter
    toolkit = CodeExecutionToolkit()
    
    # Execute code with a syntax error
    code = "print('Unclosed string"
    result = toolkit.execute_python({"code": code})
    
    # Verify the result indicates an error
    assert result["status"] == "error"
    assert result["error"] is not None
    assert "SyntaxError" in result["error"]


def test_sandboxed_vs_internal_interpreter():
    """Test the difference between sandboxed and internal interpreters."""
    # Create toolkit with the default (sandboxed) interpreter
    sandboxed_toolkit = CodeExecutionToolkit()
    
    # Create toolkit with an internal (unsafe) interpreter
    internal_interpreter = InternalPythonInterpreter()
    internal_toolkit = CodeExecutionToolkit(interpreter=internal_interpreter)
    
    # Execute simple code with both interpreters
    code = "print('Test')"
    sandboxed_result = sandboxed_toolkit.execute_python({"code": code})
    internal_result = internal_toolkit.execute_python({"code": code})
    
    # Both should work for simple code
    assert sandboxed_result["status"] == "success"
    assert internal_result["status"] == "success"
    
    # Test a more complex case where internal can access the filesystem but sandboxed might not
    # This is just a demonstration - the actual behavior depends on the implementation
    code_with_imports = "import os\nprint(os.getcwd())"
    internal_result = internal_toolkit.execute_python({"code": code_with_imports})
    assert internal_result["status"] == "success" 
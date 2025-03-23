import pytest
from unittest.mock import MagicMock

from oumi.multi_agents.api import AgentMemories


def test_agent_memories_initialization():
    """Test initializing the AgentMemories class."""
    # Test default initialization
    memory = AgentMemories()
    assert memory.get_chat_history() == []
    assert memory.max_history_length is None  # Default to no limit
    
    # Test with max history length
    memory = AgentMemories(max_history_length=5)
    assert memory.get_chat_history() == []
    assert memory.max_history_length == 5


def test_add_chat_record():
    """Test adding chat records to memory."""
    memory = AgentMemories()
    
    # Add a single record
    memory.add_chat_record(
        human_message="Hello, agent.",
        ai_message="Hello, human. How can I help you?"
    )
    
    # Verify the record was added
    history = memory.get_chat_history()
    assert len(history) == 1
    assert history[0].human_message == "Hello, agent."
    assert history[0].ai_message == "Hello, human. How can I help you?"
    
    # Add another record
    memory.add_chat_record(
        human_message="What's the weather like?",
        ai_message="I don't have real-time data, but I can help you find that information."
    )
    
    # Verify both records are present
    history = memory.get_chat_history()
    assert len(history) == 2
    assert history[1].human_message == "What's the weather like?"
    assert history[1].ai_message == "I don't have real-time data, but I can help you find that information."


def test_memory_truncation():
    """Test that memory is truncated when max_history_length is set."""
    # Create memory with max length of 2
    memory = AgentMemories(max_history_length=2)
    
    # Add three records (one more than the max)
    memory.add_chat_record("Message 1", "Response 1")
    memory.add_chat_record("Message 2", "Response 2")
    memory.add_chat_record("Message 3", "Response 3")
    
    # Verify only the most recent two are kept
    history = memory.get_chat_history()
    assert len(history) == 2
    assert history[0].human_message == "Message 2"
    assert history[0].ai_message == "Response 2"
    assert history[1].human_message == "Message 3"
    assert history[1].ai_message == "Response 3"


def test_empty_memory():
    """Test clearing memory."""
    memory = AgentMemories()
    
    # Add some records
    memory.add_chat_record("Hello", "Hi there")
    memory.add_chat_record("How are you?", "I'm fine, thanks")
    
    # Verify records were added
    assert len(memory.get_chat_history()) == 2
    
    # Clear the memory
    memory.clear()
    
    # Verify memory is empty
    assert len(memory.get_chat_history()) == 0


def test_memory_integration_with_agent():
    """Test using memory with an agent."""
    # Create mock agent
    mock_agent = MagicMock()
    mock_agent.step.return_value = MagicMock(
        msg=MagicMock(content="Agent response")
    )
    
    # Create memory
    memory = AgentMemories()
    
    # Simulate a conversation with the agent
    messages = [
        "My name is Alex.",
        "I work as a software engineer.",
        "I live in San Francisco."
    ]
    
    for message in messages:
        # Get agent response
        response = mock_agent.step(message)
        ai_message = response.msg.content
        
        # Add to memory
        memory.add_chat_record(
            human_message=message,
            ai_message=ai_message
        )
    
    # Verify all interactions are in memory
    history = memory.get_chat_history()
    assert len(history) == 3
    
    # Verify the content of the memory
    assert history[0].human_message == "My name is Alex."
    assert history[0].ai_message == "Agent response"
    assert history[1].human_message == "I work as a software engineer."
    assert history[1].ai_message == "Agent response"
    assert history[2].human_message == "I live in San Francisco."
    assert history[2].ai_message == "Agent response"


def test_serialization_deserialization():
    """Test serializing and deserializing memory."""
    original_memory = AgentMemories()
    
    # Add some chat records
    original_memory.add_chat_record("Hello", "Hi there")
    original_memory.add_chat_record("Tell me about yourself", "I'm an AI assistant")
    
    # Serialize the memory
    serialized = []
    for record in original_memory.get_chat_history():
        serialized.append({
            "human": record.human_message,
            "ai": record.ai_message,
        })
    
    # Create a new memory and deserialize
    new_memory = AgentMemories()
    for record in serialized:
        new_memory.add_chat_record(
            human_message=record["human"],
            ai_message=record["ai"],
        )
    
    # Verify the new memory has the same content
    new_history = new_memory.get_chat_history()
    original_history = original_memory.get_chat_history()
    
    assert len(new_history) == len(original_history)
    for i in range(len(new_history)):
        assert new_history[i].human_message == original_history[i].human_message
        assert new_history[i].ai_message == original_history[i].ai_message 
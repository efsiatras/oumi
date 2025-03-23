"""
Examples of using CAMEL-AI multi-agent capabilities in Oumi.

This module provides example functions to demonstrate how to use CAMEL-AI
within Oumi for various multi-agent scenarios.
"""

from typing import Dict, Any, Optional, List, Tuple, Union

from camel.agents import ChatAgent
from camel.societies import RolePlaying
from camel.types import TaskType


def simple_chat_example(
    prompt: str,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Execute a simple chat with a single agent.
    
    Args:
        prompt: The prompt to send to the agent.
        model_name: Optional model name to use.
        api_key: Optional API key for the model provider.
        
    Returns:
        str: The response from the agent.
    """
    agent = ChatAgent("You are a helpful assistant.", model=model_name)
    agent.reset()
    response = agent.step(prompt)
    return response.msg.content


def role_playing_example(
    task_prompt: str,
    assistant_role: str = "Python Programmer",
    user_role: str = "Data Scientist",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    chat_turn_limit: int = 10,
) -> List[Dict[str, str]]:
    """Execute a role-playing conversation between two agents.
    
    Args:
        task_prompt: The task to be performed by the agents.
        assistant_role: The role of the assistant agent.
        user_role: The role of the user agent.
        model_name: Optional model name to use.
        api_key: Optional API key for the model provider.
        chat_turn_limit: Maximum number of chat turns.
        
    Returns:
        List[Dict[str, str]]: A list of conversation messages.
    """
    role_play_session = RolePlaying(
        assistant_role_name=assistant_role,
        assistant_agent_kwargs=dict(model=model_name),
        user_role_name=user_role,
        user_agent_kwargs=dict(model=model_name),
        task_prompt=task_prompt,
        with_task_specify=True,
        task_specify_agent_kwargs=dict(model=model_name),
    )
    
    conversation = []
    
    # Add system messages and task prompts to conversation history
    conversation.append({
        "role": f"system ({assistant_role})",
        "content": role_play_session.assistant_sys_msg
    })
    conversation.append({
        "role": f"system ({user_role})",
        "content": role_play_session.user_sys_msg
    })
    conversation.append({
        "role": "task (original)",
        "content": task_prompt
    })
    conversation.append({
        "role": "task (specified)",
        "content": role_play_session.specified_task_prompt
    })
    
    # Initialize the chat
    input_msg = role_play_session.init_chat()
    
    # Execute the conversation turns
    n = 0
    while n < chat_turn_limit:
        n += 1
        assistant_response, user_response = role_play_session.step(input_msg)
        
        # Check for termination
        if assistant_response.terminated or user_response.terminated:
            break
            
        # Add responses to conversation history
        conversation.append({
            "role": user_role,
            "content": user_response.msg.content
        })
        conversation.append({
            "role": assistant_role,
            "content": assistant_response.msg.content
        })
        
        # Check if task is done
        if "CAMEL_TASK_DONE" in user_response.msg.content:
            break
            
        input_msg = assistant_response.msg
        
    return conversation 
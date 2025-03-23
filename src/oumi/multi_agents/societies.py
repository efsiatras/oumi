"""
CAMEL-AI societies integration for Oumi.

This module provides direct access to CAMEL-AI's societies functionality,
allowing the creation and management of multi-agent systems.
"""

from typing import Dict, Any, Optional, List, Tuple

from camel.societies import RolePlaying, BabyAGIPlaying
from camel.types import TaskType


class RolePlayingSession:
    """A wrapper around CAMEL-AI's RolePlaying society for Oumi integration."""
    
    def __init__(
        self,
        task_prompt: str,
        assistant_role_name: str,
        user_role_name: str,
        model: Optional[str] = None,
        with_task_specify: bool = True,
        chat_turn_limit: int = 50,
    ):
        """Initialize the role-playing session.
        
        Args:
            task_prompt: The task to be performed by the agents.
            assistant_role_name: The role of the assistant agent.
            user_role_name: The role of the user agent.
            model: Optional model name to use for both agents.
            with_task_specify: Whether to use task specification.
            chat_turn_limit: Maximum number of chat turns.
        """
        self.task_prompt = task_prompt
        self.assistant_role_name = assistant_role_name
        self.user_role_name = user_role_name
        self.model = model
        self.with_task_specify = with_task_specify
        self.chat_turn_limit = chat_turn_limit
        
        self.session = RolePlaying(
            assistant_role_name=assistant_role_name,
            assistant_agent_kwargs=dict(model=model),
            user_role_name=user_role_name,
            user_agent_kwargs=dict(model=model),
            task_prompt=task_prompt,
            with_task_specify=with_task_specify,
            task_specify_agent_kwargs=dict(model=model),
        )
        
        # Initialize chat
        self.current_msg = self.session.init_chat()
        self.conversation_history = []
        self._is_complete = False
        
    def execute_full_conversation(self) -> List[Dict[str, str]]:
        """Execute the conversation until completion or limit is reached.
        
        Returns:
            List[Dict[str, str]]: The full conversation history.
        """
        # Add system messages to history
        self.conversation_history.append({
            "role": f"system ({self.assistant_role_name})",
            "content": self.session.assistant_sys_msg
        })
        self.conversation_history.append({
            "role": f"system ({self.user_role_name})",
            "content": self.session.user_sys_msg
        })
        self.conversation_history.append({
            "role": "task (original)",
            "content": self.task_prompt
        })
        self.conversation_history.append({
            "role": "task (specified)",
            "content": self.session.specified_task_prompt
        })
        
        # Execute conversation turns
        n = 0
        while n < self.chat_turn_limit and not self._is_complete:
            n += 1
            self.step()
            
        return self.conversation_history
    
    def step(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Execute a single step in the conversation.
        
        Returns:
            Tuple[Dict[str, str], Dict[str, str]]: User and assistant messages.
        """
        if self._is_complete:
            return {}, {}
            
        assistant_response, user_response = self.session.step(self.current_msg)
        
        # Check for termination
        if assistant_response.terminated or user_response.terminated:
            self._is_complete = True
            
        # Add responses to conversation history
        user_message = {
            "role": self.user_role_name,
            "content": user_response.msg.content
        }
        assistant_message = {
            "role": self.assistant_role_name,
            "content": assistant_response.msg.content
        }
        
        self.conversation_history.append(user_message)
        self.conversation_history.append(assistant_message)
        
        # Check if task is done
        if "CAMEL_TASK_DONE" in user_response.msg.content:
            self._is_complete = True
            
        # Update current message for next step
        self.current_msg = assistant_response.msg
        
        return user_message, assistant_message
    
    def is_complete(self) -> bool:
        """Check if the conversation is complete.
        
        Returns:
            bool: True if conversation is complete, False otherwise.
        """
        return self._is_complete
        
    @property
    def system_messages(self) -> Dict[str, str]:
        """Get the system messages for the session.
        
        Returns:
            Dict[str, str]: The system messages.
        """
        return {
            "assistant": self.session.assistant_sys_msg,
            "user": self.session.user_sys_msg,
            "task_original": self.task_prompt,
            "task_specified": self.session.specified_task_prompt,
        } 
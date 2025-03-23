"""
Oumi integration with CAMEL-AI multi-agent framework.

This module provides direct access to CAMEL-AI's multi-agent capabilities
through a modular and maintainable interface.

Note: This module requires the camel-ai package to be installed.
"""

# Re-export the main components
from camel.agents import (
    ChatAgent,
    CriticAgent,
    DeductiveReasonerAgent,
    EmbodiedAgent,
    KnowledgeGraphAgent,
    RoleAssignmentAgent,
    SearchAgent,
    TaskAgent,
)
from camel.societies import RolePlaying, BabyAGIPlaying
from camel.models import ModelFactory
from camel.types import TaskType, RoleType

# Import local modules
from oumi.multi_agents.utils import (
    get_available_models,
    create_camel_model,
    get_camel_version,
)
from oumi.multi_agents.toolkits import (
    get_available_toolkits,
    create_toolkit,
)
from oumi.multi_agents.societies import RolePlayingSession
from oumi.multi_agents.adapters import (
    OumiModelAdapter,
    create_agent_model,
)

# The full API surface is available through the api module
from oumi.multi_agents.api import *  # noqa

# Define what should be accessible when `from oumi.multi_agents import *` is used
__all__ = [
    # Agents from camel
    "ChatAgent",
    "CriticAgent",
    "DeductiveReasonerAgent",
    "EmbodiedAgent",
    "KnowledgeGraphAgent",
    "RoleAssignmentAgent", 
    "SearchAgent",
    "TaskAgent",
    
    # Societies from camel
    "RolePlaying",
    "BabyAGIPlaying",
    
    # Utilities from camel
    "ModelFactory",
    "TaskType",
    "RoleType",
    
    # Custom utilities
    "get_available_models",
    "create_camel_model",
    "get_camel_version",
    "get_available_toolkits",
    "create_toolkit",
    "RolePlayingSession",
    
    # Adapters
    "OumiModelAdapter",
    "create_agent_model",
] 
"""
CAMEL-AI toolkits integration for Oumi.

This module provides direct access to CAMEL-AI's various toolkits
that can be used by agents for different specialized tasks.
"""

from typing import Dict, Any, Optional, List, Union

from camel.toolkits import (
    BaseToolkit,
    CodeExecutionToolkit,
    DalleToolkit,
    GitHubToolkit,
    GoogleMapsToolkit,
    LinkedInToolkit,
    MathToolkit,
    OpenAPIToolkit,
    OpenAIFunctionToolkit,
    RedditToolkit,
    RetrievalToolkit,
    SearchToolkit,
    SlackToolkit,
    TwitterToolkit,
    WeatherToolkit,
)


def get_available_toolkits() -> List[str]:
    """Return a list of all available CAMEL toolkits.
    
    Returns:
        List[str]: A list of toolkit names.
    """
    return [
        "CodeExecutionToolkit",
        "DalleToolkit",
        "GitHubToolkit",
        "GoogleMapsToolkit",
        "LinkedInToolkit",
        "MathToolkit",
        "OpenAPIToolkit",
        "OpenAIFunctionToolkit",
        "RedditToolkit",
        "RetrievalToolkit",
        "SearchToolkit",
        "SlackToolkit",
        "TwitterToolkit",
        "WeatherToolkit",
    ]


def create_toolkit(
    toolkit_name: str,
    api_key: Optional[str] = None,
    **toolkit_kwargs: Any,
) -> BaseToolkit:
    """Create a CAMEL toolkit instance.
    
    Args:
        toolkit_name: The name of the toolkit to create.
        api_key: Optional API key for the toolkit service.
        **toolkit_kwargs: Additional keyword arguments for the toolkit.
        
    Returns:
        BaseToolkit: A CAMEL toolkit instance.
        
    Raises:
        ValueError: If the toolkit name is not recognized.
    """
    toolkit_classes = {
        "CodeExecutionToolkit": CodeExecutionToolkit,
        "DalleToolkit": DalleToolkit,
        "GitHubToolkit": GitHubToolkit,
        "GoogleMapsToolkit": GoogleMapsToolkit,
        "LinkedInToolkit": LinkedInToolkit,
        "MathToolkit": MathToolkit,
        "OpenAPIToolkit": OpenAPIToolkit,
        "OpenAIFunctionToolkit": OpenAIFunctionToolkit,
        "RedditToolkit": RedditToolkit,
        "RetrievalToolkit": RetrievalToolkit,
        "SearchToolkit": SearchToolkit,
        "SlackToolkit": SlackToolkit,
        "TwitterToolkit": TwitterToolkit,
        "WeatherToolkit": WeatherToolkit,
    }
    
    if toolkit_name not in toolkit_classes:
        raise ValueError(
            f"Unknown toolkit: {toolkit_name}. "
            f"Available toolkits: {list(toolkit_classes.keys())}"
        )
    
    toolkit_class = toolkit_classes[toolkit_name]
    
    # Adjust args based on toolkit requirements
    if toolkit_name in ["DalleToolkit", "OpenAIFunctionToolkit"]:
        return toolkit_class(api_key=api_key, **toolkit_kwargs)
    else:
        return toolkit_class(**toolkit_kwargs) 
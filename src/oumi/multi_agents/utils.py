"""
Utility functions for integrating CAMEL-AI with Oumi.

This module provides helper functions to facilitate the use of CAMEL-AI's
multi-agent capabilities within the Oumi framework.
"""

from typing import Dict, Any, Optional, List, Union

import camel
from camel.models import ModelFactory


def get_available_models() -> List[str]:
    """Return a list of all available models that can be used with CAMEL-AI.
    
    Returns:
        List[str]: A list of model identifiers.
    """
    return ModelFactory.get_available_models()


def create_camel_model(
    model_name: str, 
    api_key: Optional[str] = None,
    **model_kwargs: Any,
) -> Any:
    """Create a CAMEL-AI model instance.
    
    Args:
        model_name: The name of the model to use.
        api_key: Optional API key for the model provider.
        **model_kwargs: Additional keyword arguments for the model.
        
    Returns:
        Any: A CAMEL-AI model instance.
    """
    return ModelFactory.create(model_name, api_key=api_key, **model_kwargs)


def get_camel_version() -> str:
    """Return the current version of CAMEL-AI being used.
    
    Returns:
        str: The version string of the installed CAMEL-AI package.
    """
    return camel.__version__ 
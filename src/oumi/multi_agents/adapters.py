"""
Adapters to use Oumi's inference engines with CAMEL-AI.

This module provides adapter classes that allow Oumi's inference engines
to be used as model backends for CAMEL agents.
"""

from typing import Any, Dict, List, Optional, Union

import camel
from camel.models.base_model import BaseModelBackend
from camel.types import ModelMessage, ModelType, RoleType

from oumi.inference import (
    AnthropicInferenceEngine,
    LlamaCppInferenceEngine,
    NativeTextInferenceEngine,
    OpenAIInferenceEngine,
    VLLMInferenceEngine,
)


class OumiModelAdapter(BaseModelBackend):
    """Adapter that allows Oumi inference engines to be used with CAMEL.
    
    This adapter wraps an Oumi inference engine and implements the CAMEL
    model interface so that Oumi's models can be used with CAMEL agents.
    """
    
    def __init__(
        self,
        inference_engine: Any,
        model_type: ModelType = ModelType.GPT_3_5_TURBO,
        system_message: Optional[str] = None,
    ):
        """Initialize the Oumi model adapter.
        
        Args:
            inference_engine: An Oumi inference engine instance.
            model_type: The CAMEL model type to use.
            system_message: Optional system message for the model.
        """
        super().__init__(model_type=model_type)
        self.inference_engine = inference_engine
        self.system_message = system_message
        
    def run(
        self,
        messages: List[ModelMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Run the model with the given messages.
        
        Args:
            messages: A list of CAMEL model messages.
            temperature: The temperature for sampling.
            max_tokens: The maximum number of tokens to generate.
            **kwargs: Additional keyword arguments for the inference engine.
            
        Returns:
            str: The model's response text.
        """
        # Convert CAMEL messages to Oumi format
        oumi_messages = []
        
        # Add system message if provided
        if self.system_message:
            oumi_messages.append({"role": "system", "content": self.system_message})
        
        # Add the other messages
        for msg in messages:
            role = "user" if msg.role == RoleType.USER else "assistant"
            oumi_messages.append({"role": role, "content": msg.content})
        
        # Run inference
        response = self.inference_engine.chat_completion(
            messages=oumi_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        
        # Extract and return the response content
        return response.choices[0].message.content


def create_agent_model(
    model_name_or_path: str,
    model_type: Optional[str] = None,
    system_message: Optional[str] = None,
    **kwargs,
) -> OumiModelAdapter:
    """Create an Oumi model adapter for use with multi-agent systems with CAMEL.
    
    This is a factory function that creates the appropriate inference engine
    and wraps it in an adapter for use with the multi-agent framework CAMEL.
    
    Args:
        model_name_or_path: The name or path of the model to use.
        model_type: The type of the model (openai, llama, etc.).
        system_message: Optional system message for the model.
        **kwargs: Additional arguments for the inference engine.
        
    Returns:
        OumiModelAdapter: An adapter that can be used with the multi-agent framework.
        
    Raises:
        ValueError: If the model type is not recognized.
    """
    # Determine the model type if not provided
    if not model_type:
        if "gpt" in model_name_or_path.lower():
            model_type = "openai"
        elif "llama" in model_name_or_path.lower():
            model_type = "llama"
        elif "claude" in model_name_or_path.lower():
            model_type = "anthropic"
        else:
            model_type = "native"  # Default to native
    
    # Create the appropriate inference engine
    if model_type == "openai":
        engine = OpenAIInferenceEngine(model=model_name_or_path, **kwargs)
        camel_model_type = ModelType.GPT_4
    elif model_type == "llama":
        engine = LlamaCppInferenceEngine(model_path=model_name_or_path, **kwargs)
        camel_model_type = ModelType.STUB
    elif model_type == "anthropic":
        engine = AnthropicInferenceEngine(model=model_name_or_path, **kwargs)
        camel_model_type = ModelType.CLAUDE
    elif model_type == "vllm":
        engine = VLLMInferenceEngine(model=model_name_or_path, **kwargs)
        camel_model_type = ModelType.STUB
    elif model_type == "native":
        engine = NativeTextInferenceEngine(model_path=model_name_or_path, **kwargs)
        camel_model_type = ModelType.STUB
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create and return the adapter
    return OumiModelAdapter(
        inference_engine=engine,
        model_type=camel_model_type,
        system_message=system_message,
    ) 
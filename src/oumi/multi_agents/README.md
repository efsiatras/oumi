# Oumi Multi-Agent System (powered by CAMEL-AI)

This module integrates the [CAMEL-AI](https://github.com/camel-ai/camel) framework into Oumi, providing robust multi-agent capabilities.

## Overview

CAMEL-AI is a framework that enables the creation and coordination of multiple AI agents to collaborate on tasks. This integration makes CAMEL's capabilities available as a native feature within Oumi.

## Key Features

- **Direct API Access**: Use CAMEL-AI APIs directly without wrappers or reimplementation
- **Multi-Agent Support**: Create agent societies with predefined roles and tasks
- **Tool Integration**: Access to various specialized toolkits for different tasks
- **Modular Design**: Clean integration with minimal dependencies
- **Oumi Model Support**: Use Oumi's inference engines as models for CAMEL agents
- **CLI Access**: Command-line interface for multi-agent capabilities
- **Complete API Surface**: Access to all CAMEL-AI functionality

## Using with Oumi Models

You can use Oumi's inference engines with CAMEL agents:

```python
from oumi.multi_agents import create_agent_model, ChatAgent

# Create an Oumi model adapter for CAMEL
oumi_model = create_agent_model(
    model_name_or_path="your_model_path",
    model_type="llama",  # or "openai", "anthropic", "vllm", "native"
)

# Use it with a CAMEL agent
agent = ChatAgent("You are a helpful assistant.", model=oumi_model)
agent.reset()
response = agent.step("What is machine learning?")
print(response.msg.content)
```

## CLI Usage

The Oumi CLI now includes multi-agent commands:

```bash
# Simple chat with an agent
oumi multi-agents chat "What is reinforcement learning?"

# Role-playing with two agents
oumi multi-agents role-play "Design a recommendation system for an e-commerce website" \
    --assistant "Machine Learning Engineer" \
    --user "E-commerce Manager"

# List available models
oumi multi-agents list-models

# List available toolkits
oumi multi-agents list-toolkits

# Demonstrate a toolkit
oumi multi-agents toolkit-demo "WeatherToolkit" "What's the weather in San Francisco?"
```

## Python API Examples

### Simple Single Agent

```python
from oumi.multi_agents import ChatAgent

agent = ChatAgent("You are a helpful assistant.")
agent.reset()
response = agent.step("What is machine learning?")
print(response.msg.content)
```

### Role Playing with Two Agents

```python
from oumi.multi_agents import RolePlayingSession

session = RolePlayingSession(
    task_prompt="Design a trading strategy for cryptocurrency markets",
    assistant_role_name="Financial Analyst",
    user_role_name="Software Developer"
)

conversation = session.execute_full_conversation()
for message in conversation:
    print(f"{message['role']}: {message['content']}\n")
```

### Using Toolkits

```python
from oumi.multi_agents import create_toolkit, ChatAgent
from oumi.multi_agents.toolkits import get_available_toolkits

# List available toolkits
print(get_available_toolkits())

# Create a toolkit
weather_toolkit = create_toolkit("WeatherToolkit")

# Use toolkit with an agent
agent = ChatAgent("You are a helpful assistant.")
agent.add_toolkit(weather_toolkit)
```

### Advanced Functionality

```python
# Importing from the full API surface
from oumi.multi_agents.api import (
    BaseAgent,
    OpenAIModel,
    VectorRetriever,
    CoTDataGenerator,
    TokenLimitTerminator,
)

# Access to all CAMEL functionality
from oumi.multi_agents.api import *
```

## Available Components

- **Agents**: ChatAgent, CriticAgent, TaskAgent, and more
- **Societies**: RolePlaying, BabyAGIPlaying
- **Toolkits**: Various specialized tools for different tasks
- **Adapters**: OumiModelAdapter for using Oumi models with CAMEL
- **CLI Commands**: Direct access through the Oumi CLI
- **Complete API Surface**: All CAMEL modules and classes

## Module Structure

- `__init__.py` - Main entry point with re-exports of key components
- `adapters.py` - Adapter for Oumi models to work with CAMEL
- `api.py` - Complete CAMEL API surface
- `examples.py` - Example implementations
- `societies.py` - Helper classes for agent societies
- `toolkits.py` - Access to CAMEL toolkits
- `utils.py` - Utility functions

## Requirements

The camel-ai package is required and will be installed as a dependency of Oumi:

```bash
pip install camel-ai
```

## Documentation

For complete documentation on the CAMEL-AI framework, refer to the [official CAMEL-AI documentation](https://docs.camel-ai.org/). 
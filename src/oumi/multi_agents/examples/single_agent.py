"""
Example showing how to use a single CAMEL agent with Oumi.

This example demonstrates how to create and use a single agent to respond to a prompt.
"""

import argparse
from typing import Optional

from oumi.multi_agents import ChatAgent
from oumi.multi_agents.adapters import create_agent_model


def main(
    prompt: str = "Explain quantum computing in simple terms",
    oumi_model_path: Optional[str] = None,
    camel_model_name: Optional[str] = None,
    temperature: float = 0.7,
):
    """Run a simple chat with a single agent.

    Args:
        prompt: The prompt to send to the agent.
        oumi_model_path: Optional path to an Oumi model.
        camel_model_name: Optional name of a CAMEL model.
        temperature: Temperature for sampling.
    """
    print(f"Running chat with prompt: {prompt}")
    
    # Decide which model to use
    model = None
    if oumi_model_path:
        print(f"Using Oumi model: {oumi_model_path}")
        model = create_agent_model(
            model_name_or_path=oumi_model_path,
            system_message="You are a helpful assistant."
        )
    elif camel_model_name:
        print(f"Using CAMEL model: {camel_model_name}")
        model = camel_model_name
    
    # Create and use the agent
    agent = ChatAgent("You are a helpful assistant.", model=model)
    agent.reset()
    response = agent.step(prompt)
    
    print("\nResponse:")
    print(response.msg.content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a simple chat with a single agent")
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="Explain quantum computing in simple terms",
        help="Prompt to send to the agent"
    )
    parser.add_argument(
        "--oumi-model", 
        type=str, 
        help="Path to an Oumi model"
    )
    parser.add_argument(
        "--camel-model", 
        type=str,
        help="Name of a CAMEL model"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Temperature for sampling"
    )
    
    args = parser.parse_args()
    main(
        prompt=args.prompt,
        oumi_model_path=args.oumi_model,
        camel_model_name=args.camel_model,
        temperature=args.temperature,
    ) 
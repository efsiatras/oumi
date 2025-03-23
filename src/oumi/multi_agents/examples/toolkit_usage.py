"""
Example showing how to use CAMEL toolkits with Oumi.

This example demonstrates how to create and use various toolkits with agents.
"""

import argparse
import os
from typing import Optional

from rich.console import Console

from oumi.multi_agents import ChatAgent
from oumi.multi_agents.adapters import create_agent_model
from oumi.multi_agents.toolkits import create_toolkit, get_available_toolkits


console = Console()


def main(
    toolkit_name: str = "MathToolkit",
    query: str = "Calculate the integral of x^2 from 0 to 1",
    oumi_model_path: Optional[str] = None,
    camel_model_name: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """Run a chat with an agent using a specific toolkit.

    Args:
        toolkit_name: The name of the toolkit to use.
        query: The query to send to the agent.
        oumi_model_path: Optional path to an Oumi model.
        camel_model_name: Optional name of a CAMEL model.
        api_key: Optional API key for toolkit services.
    """
    # Check if the toolkit is available
    available_toolkits = get_available_toolkits()
    if toolkit_name not in available_toolkits:
        console.print(f"[bold red]Error:[/bold red] Toolkit '{toolkit_name}' not found.")
        console.print("[bold blue]Available toolkits:[/bold blue]")
        for tk in available_toolkits:
            console.print(f"- {tk}")
        return
    
    console.print(f"[bold green]Using toolkit:[/bold green] {toolkit_name}")
    console.print(f"[bold green]Query:[/bold green] {query}")
    
    # Use provided API key or try to get from environment variables
    if not api_key:
        if toolkit_name == "OpenAIFunctionToolkit" or toolkit_name == "DalleToolkit":
            api_key = os.environ.get("OPENAI_API_KEY")
            console.print("[bold yellow]Using OPENAI_API_KEY from environment variables[/bold yellow]")
        elif toolkit_name == "WeatherToolkit":
            api_key = os.environ.get("WEATHER_API_KEY")
            console.print("[bold yellow]Using WEATHER_API_KEY from environment variables[/bold yellow]")
    
    # Create the toolkit
    try:
        toolkit = create_toolkit(toolkit_name, api_key=api_key)
    except ValueError as e:
        console.print(f"[bold red]Error creating toolkit:[/bold red] {e}")
        return
    
    # Decide which model to use
    model = None
    if oumi_model_path:
        console.print(f"[bold yellow]Using Oumi model:[/bold yellow] {oumi_model_path}")
        model = create_agent_model(model_name_or_path=oumi_model_path)
    elif camel_model_name:
        console.print(f"[bold yellow]Using CAMEL model:[/bold yellow] {camel_model_name}")
        model = camel_model_name
    
    # Create and use the agent with the toolkit
    agent = ChatAgent("You are a helpful assistant.", model=model)
    agent.add_toolkit(toolkit)
    agent.reset()
    
    console.print("[bold]Sending query to agent with toolkit...[/bold]")
    response = agent.step(query)
    
    console.print("\n[bold]Response:[/bold]")
    console.print(response.msg.content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a chat with an agent using a specific toolkit")
    parser.add_argument(
        "--toolkit",
        type=str,
        default="MathToolkit",
        help="Name of the toolkit to use"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Calculate the integral of x^2 from 0 to 1",
        help="Query to send to the agent"
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
        "--api-key",
        type=str,
        help="API key for toolkit services"
    )
    
    args = parser.parse_args()
    main(
        toolkit_name=args.toolkit,
        query=args.query,
        oumi_model_path=args.oumi_model,
        camel_model_name=args.camel_model,
        api_key=args.api_key,
    ) 
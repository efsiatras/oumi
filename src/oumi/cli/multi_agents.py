"""
CLI commands for CAMEL multi-agent functionality.

This module provides CLI commands for using CAMEL multi-agent capabilities
in Oumi.
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import typer
from rich import print as rprint

from oumi.cli.cli_utils import CONSOLE, validate_file_exists
from oumi.multi_agents import (
    ChatAgent,
    RolePlayingSession,
    get_available_models,
    get_available_toolkits,
)
from oumi.multi_agents.adapters import create_agent_model


app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def chat(
    prompt: str = typer.Argument(..., help="The prompt to send to the agent."),
    model: str = typer.Option(
        None, "--model", "-m", help="Model to use for the agent."
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="File to save the response to."
    ),
    temperature: float = typer.Option(
        0.7, "--temperature", "-t", help="Temperature for sampling."
    ),
):
    """Run a simple chat with a single agent."""
    CONSOLE.print(f"Running chat with prompt: [bold green]{prompt}[/bold green]")
    
    agent = ChatAgent("You are a helpful assistant.", model=model)
    agent.reset()
    response = agent.step(prompt)
    
    CONSOLE.print("\n[bold]Response:[/bold]")
    CONSOLE.print(response.msg.content)
    
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(response.msg.content)
        CONSOLE.print(f"\nResponse saved to: [bold blue]{output_file}[/bold blue]")


@app.command()
def role_play(
    task: str = typer.Argument(..., help="The task for the agents to perform."),
    assistant_role: str = typer.Option(
        "Python Programmer", "--assistant", "-a", help="Role of the assistant agent."
    ),
    user_role: str = typer.Option(
        "Product Manager", "--user", "-u", help="Role of the user agent."
    ),
    model: str = typer.Option(
        None, "--model", "-m", help="Model to use for the agents."
    ),
    oumi_model_path: Optional[str] = typer.Option(
        None,
        "--oumi-model",
        help="Oumi model path to use instead of a CAMEL model.",
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="File to save the conversation to."
    ),
    turns: int = typer.Option(
        10, "--turns", "-n", help="Maximum number of conversation turns."
    ),
):
    """Run a role-playing conversation between two agents."""
    CONSOLE.print(f"Starting role-play conversation for task: [bold green]{task}[/bold green]")
    CONSOLE.print(f"Assistant role: [bold blue]{assistant_role}[/bold blue]")
    CONSOLE.print(f"User role: [bold blue]{user_role}[/bold blue]")
    
    # Use Oumi model if specified
    if oumi_model_path:
        model = create_agent_model(model_name_or_path=oumi_model_path)
    
    # Create and run the role-playing session
    session = RolePlayingSession(
        task_prompt=task,
        assistant_role_name=assistant_role,
        user_role_name=user_role,
        model=model,
        chat_turn_limit=turns,
    )
    
    conversation = session.execute_full_conversation()
    
    # Display conversation
    CONSOLE.print("\n[bold]Conversation:[/bold]")
    for msg in conversation:
        role = msg["role"]
        content = msg["content"]
        
        if "system" in role:
            CONSOLE.print(f"[bold blue]{role}:[/bold blue]")
            CONSOLE.print(f"{content}\n")
        elif "task" in role:
            CONSOLE.print(f"[bold yellow]{role}:[/bold yellow]")
            CONSOLE.print(f"{content}\n")
        else:
            CONSOLE.print(f"[bold green]{role}:[/bold green]")
            CONSOLE.print(f"{content}\n")
    
    # Save to file if specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(conversation, f, indent=2)
        CONSOLE.print(f"\nConversation saved to: [bold blue]{output_file}[/bold blue]")


@app.command()
def list_models():
    """List available CAMEL models."""
    models = get_available_models()
    CONSOLE.print("[bold]Available CAMEL models:[/bold]")
    for model in models:
        CONSOLE.print(f"- {model}")


@app.command()
def list_toolkits():
    """List available CAMEL toolkits."""
    toolkits = get_available_toolkits()
    CONSOLE.print("[bold]Available CAMEL toolkits:[/bold]")
    for toolkit in toolkits:
        CONSOLE.print(f"- {toolkit}")


@app.command()
def toolkit_demo(
    toolkit_name: str = typer.Argument(
        ..., help="The name of the toolkit to demonstrate."
    ),
    query: str = typer.Argument(
        ..., help="The query to use for the demonstration."
    ),
    model: str = typer.Option(
        None, "--model", "-m", help="Model to use for the agent."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key for toolkit services."
    ),
):
    """Demonstrate a CAMEL toolkit with a simple query."""
    from oumi.multi_agents.toolkits import create_toolkit
    
    CONSOLE.print(f"Demonstrating toolkit: [bold green]{toolkit_name}[/bold green]")
    CONSOLE.print(f"Query: [bold blue]{query}[/bold blue]")
    
    # Create toolkit
    try:
        toolkit = create_toolkit(toolkit_name, api_key=api_key)
    except ValueError as e:
        CONSOLE.print(f"[bold red]Error:[/bold red] {e}")
        return
    
    # Create agent with toolkit
    agent = ChatAgent("You are a helpful assistant.", model=model)
    agent.add_toolkit(toolkit)
    agent.reset()
    
    # Run query
    response = agent.step(query)
    
    CONSOLE.print("\n[bold]Response:[/bold]")
    CONSOLE.print(response.msg.content) 
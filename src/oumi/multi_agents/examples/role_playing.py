"""
Example showing how to use role-playing with two CAMEL agents in Oumi.

This example demonstrates how to set up a role-playing conversation
between two agents with different roles.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from rich.console import Console

from oumi.multi_agents import RolePlayingSession
from oumi.multi_agents.adapters import create_agent_model


console = Console()


def main(
    task_prompt: str = "Design an algorithm for sentiment analysis of social media posts",
    assistant_role: str = "Machine Learning Engineer",
    user_role: str = "Product Manager",
    oumi_model_path: Optional[str] = None,
    camel_model_name: Optional[str] = None,
    output_file: Optional[str] = None,
    chat_turn_limit: int = 10,
):
    """Run a role-playing conversation between two agents.

    Args:
        task_prompt: The task to be performed by the agents.
        assistant_role: The role of the assistant agent.
        user_role: The role of the user agent.
        oumi_model_path: Optional path to an Oumi model.
        camel_model_name: Optional name of a CAMEL model.
        output_file: Optional file to save the conversation to.
        chat_turn_limit: Maximum number of conversation turns.
    """
    console.print(f"[bold green]Task:[/bold green] {task_prompt}")
    console.print(f"[bold blue]Assistant role:[/bold blue] {assistant_role}")
    console.print(f"[bold blue]User role:[/bold blue] {user_role}")
    
    # Decide which model to use
    model = None
    if oumi_model_path:
        console.print(f"[bold yellow]Using Oumi model:[/bold yellow] {oumi_model_path}")
        model = create_agent_model(model_name_or_path=oumi_model_path)
    elif camel_model_name:
        console.print(f"[bold yellow]Using CAMEL model:[/bold yellow] {camel_model_name}")
        model = camel_model_name
    
    # Create the role-playing session
    session = RolePlayingSession(
        task_prompt=task_prompt,
        assistant_role_name=assistant_role,
        user_role_name=user_role,
        model=model,
        chat_turn_limit=chat_turn_limit,
    )
    
    # Execute the conversation
    conversation = session.execute_full_conversation()
    
    # Display the conversation
    console.print("\n[bold]Conversation:[/bold]")
    for msg in conversation:
        role = msg["role"]
        content = msg["content"]
        
        if "system" in role:
            console.print(f"[bold blue]{role}:[/bold blue]")
            console.print(f"{content}\n")
        elif "task" in role:
            console.print(f"[bold yellow]{role}:[/bold yellow]")
            console.print(f"{content}\n")
        else:
            console.print(f"[bold green]{role}:[/bold green]")
            console.print(f"{content}\n")
    
    # Save to file if specified
    if output_file:
        output_path = Path(output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(conversation, f, indent=2)
        console.print(f"\nConversation saved to: [bold blue]{output_path}[/bold blue]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a role-playing conversation between two agents")
    parser.add_argument(
        "--task",
        type=str,
        default="Design an algorithm for sentiment analysis of social media posts",
        help="Task for the agents to perform"
    )
    parser.add_argument(
        "--assistant",
        type=str,
        default="Machine Learning Engineer",
        help="Role of the assistant agent"
    )
    parser.add_argument(
        "--user",
        type=str,
        default="Product Manager",
        help="Role of the user agent"
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
        "--output",
        type=str,
        help="File to save the conversation to"
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=10,
        help="Maximum number of conversation turns"
    )
    
    args = parser.parse_args()
    main(
        task_prompt=args.task,
        assistant_role=args.assistant,
        user_role=args.user,
        oumi_model_path=args.oumi_model,
        camel_model_name=args.camel_model,
        output_file=args.output,
        chat_turn_limit=args.turns,
    ) 
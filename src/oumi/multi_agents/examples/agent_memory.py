"""
Example showing how to use CAMEL's memory capabilities with Oumi.

This example demonstrates how to create agents with memory that persists
between interactions.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, List

from rich.console import Console

from oumi.multi_agents import ChatAgent
from oumi.multi_agents.adapters import create_agent_model
from oumi.multi_agents.api import AgentMemories


console = Console()


def main(
    conversation: List[str] = None,
    oumi_model_path: Optional[str] = None,
    camel_model_name: Optional[str] = None,
    memory_file: Optional[str] = None,
):
    """Run a chat with an agent that has memory of past interactions.

    Args:
        conversation: List of user messages for the conversation.
        oumi_model_path: Optional path to an Oumi model.
        camel_model_name: Optional name of a CAMEL model.
        memory_file: Optional file to save/load the agent's memory.
    """
    # Default conversation if none provided
    if conversation is None:
        conversation = [
            "My name is Alex.",
            "I work as a data scientist.",
            "What was my name again?",
            "What do I do for a living?",
            "I also like hiking on weekends.",
            "What are my hobbies?",
        ]
    
    # Decide which model to use
    model = None
    if oumi_model_path:
        console.print(f"[bold yellow]Using Oumi model:[/bold yellow] {oumi_model_path}")
        model = create_agent_model(model_name_or_path=oumi_model_path)
    elif camel_model_name:
        console.print(f"[bold yellow]Using CAMEL model:[/bold yellow] {camel_model_name}")
        model = camel_model_name
    
    # Create the agent with memory
    system_message = (
        "You are a helpful assistant with memory. "
        "Remember important details about the user."
    )
    agent = ChatAgent(system_message, model=model)
    
    # Create memory for the agent
    memory = AgentMemories()
    
    # Load memory from file if provided and exists
    if memory_file and Path(memory_file).exists():
        try:
            with open(memory_file, "r", encoding="utf-8") as f:
                memory_data = json.load(f)
                for record in memory_data:
                    memory.add_chat_record(
                        human_message=record["human"],
                        ai_message=record["ai"],
                    )
            console.print(f"[bold green]Loaded memory from:[/bold green] {memory_file}")
        except Exception as e:
            console.print(f"[bold red]Error loading memory:[/bold red] {e}")
    
    # Reset the agent
    agent.reset()
    
    # Process the conversation with memory
    console.print("[bold]Starting conversation with memory:[/bold]\n")
    
    for i, message in enumerate(conversation):
        console.print(f"[bold blue]User ({i+1}):[/bold blue] {message}")
        
        # Get agent response
        response = agent.step(message)
        console.print(f"[bold green]Assistant:[/bold green] {response.msg.content}\n")
        
        # Add to memory
        memory.add_chat_record(
            human_message=message,
            ai_message=response.msg.content,
        )
    
    # Save memory to file if provided
    if memory_file:
        memory_data = []
        for record in memory.get_chat_history():
            memory_data.append({
                "human": record.human_message,
                "ai": record.ai_message,
            })
        
        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, indent=2)
        
        console.print(f"[bold green]Saved memory to:[/bold green] {memory_file}")
    
    # Show memory summary
    console.print("[bold]Memory summary:[/bold]")
    for i, record in enumerate(memory.get_chat_history()):
        console.print(f"[bold]Interaction {i+1}:[/bold]")
        console.print(f"[bold blue]User:[/bold blue] {record.human_message}")
        console.print(f"[bold green]Assistant:[/bold green] {record.ai_message}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a chat with an agent that has memory")
    parser.add_argument(
        "--messages",
        nargs="+",
        help="List of messages for the conversation"
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
        "--memory-file",
        type=str,
        help="File to save/load the agent's memory"
    )
    
    args = parser.parse_args()
    main(
        conversation=args.messages,
        oumi_model_path=args.oumi_model,
        camel_model_name=args.camel_model,
        memory_file=args.memory_file,
    ) 
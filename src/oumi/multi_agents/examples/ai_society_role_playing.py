"""
Example showing how to use CAMEL's AI Society role-playing with Oumi.

This example demonstrates how to create more complex role-playing scenarios
with AI society roles and customized tasks.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

from rich.console import Console

from oumi.multi_agents import RolePlaying
from oumi.multi_agents.adapters import create_agent_model
from oumi.multi_agents.api import (
    TaskPrompt,
    Task,
    PromptTemplateGenerator,
    TaskType,
    DEFAULT_ROLE_PROMPT_TEMPLATE,
    GENERAL_ASSISTANT_ROLES,
    GENERAL_USER_ROLES,
)


console = Console()


def generate_custom_task(assistant_role: str, user_role: str, task_type: str) -> Task:
    """Generate a custom task based on roles and task type.
    
    Args:
        assistant_role: The role of the assistant agent.
        user_role: The role of the user agent.
        task_type: The type of task to generate.
        
    Returns:
        Task: A CAMEL Task object.
    """
    # Initialize the prompt template generator
    prompt_generator = PromptTemplateGenerator()
    
    # Convert string task type to enum
    task_type_enum = TaskType.AI_SOCIETY
    if task_type.upper() == "CODE":
        task_type_enum = TaskType.CODE
    elif task_type.upper() == "MISALIGNMENT":
        task_type_enum = TaskType.MISALIGNMENT
    
    # Get task specification prompt
    task_prompt_template = prompt_generator.get_task_specify_prompt(task_type_enum)
    
    # Create roles prompt
    role_prompt = DEFAULT_ROLE_PROMPT_TEMPLATE.format(
        assistant_role=assistant_role,
        user_role=user_role,
    )
    
    # Create task prompt
    task_prompt = TaskPrompt(
        system_prompt=task_prompt_template.get_system_prompt(),
        content=task_prompt_template.format(role_prompt=role_prompt),
    )
    
    # Create and return task
    return Task(task_prompt=task_prompt)


def list_available_roles(role_type: str = "all") -> None:
    """List available pre-defined roles.
    
    Args:
        role_type: Type of roles to list ('assistant', 'user', or 'all').
    """
    if role_type.lower() in ["assistant", "all"]:
        console.print("[bold blue]Available Assistant Roles:[/bold blue]")
        for role in sorted(GENERAL_ASSISTANT_ROLES):
            console.print(f"- {role}")
        console.print()
        
    if role_type.lower() in ["user", "all"]:
        console.print("[bold green]Available User Roles:[/bold green]")
        for role in sorted(GENERAL_USER_ROLES):
            console.print(f"- {role}")


def main(
    assistant_role: str = "Python Programmer",
    user_role: str = "Product Manager",
    task_prompt: Optional[str] = None,
    task_type: str = "ai_society",
    oumi_model_path: Optional[str] = None,
    camel_model_name: Optional[str] = None,
    output_file: Optional[str] = None,
    custom_task: bool = False,
    chat_turn_limit: int = 15,
):
    """Run an AI Society role-playing conversation.

    Args:
        assistant_role: The role of the assistant agent.
        user_role: The role of the user agent.
        task_prompt: Optional specific task prompt.
        task_type: Type of task to generate.
        oumi_model_path: Optional path to an Oumi model.
        camel_model_name: Optional name of a CAMEL model.
        output_file: Optional file to save the conversation to.
        custom_task: Whether to generate a custom task.
        chat_turn_limit: Maximum number of conversation turns.
    """
    console.print(f"[bold blue]Assistant role:[/bold blue] {assistant_role}")
    console.print(f"[bold green]User role:[/bold green] {user_role}")
    
    # Decide which model to use
    model = None
    if oumi_model_path:
        console.print(f"[bold yellow]Using Oumi model:[/bold yellow] {oumi_model_path}")
        model = create_agent_model(model_name_or_path=oumi_model_path)
    elif camel_model_name:
        console.print(f"[bold yellow]Using CAMEL model:[/bold yellow] {camel_model_name}")
        model = camel_model_name
    
    # Set up role-playing
    role_play_kwargs: Dict[str, Any] = {
        "assistant_role_name": assistant_role,
        "assistant_agent_kwargs": dict(model=model),
        "user_role_name": user_role,
        "user_agent_kwargs": dict(model=model),
        "with_task_specify": True,
        "task_specify_agent_kwargs": dict(model=model),
    }
    
    # Handle task prompt or custom task generation
    if custom_task:
        console.print(f"[bold yellow]Generating custom {task_type} task...[/bold yellow]")
        custom_task_obj = generate_custom_task(assistant_role, user_role, task_type)
        role_play_kwargs["task"] = custom_task_obj
        role_play_kwargs["with_task_specify"] = False
    elif task_prompt:
        console.print(f"[bold yellow]Using provided task:[/bold yellow] {task_prompt}")
        role_play_kwargs["task_prompt"] = task_prompt
    else:
        # Default task if none provided
        default_task = (
            f"Collaborate on developing a solution where the {assistant_role} "
            f"helps the {user_role} solve a relevant problem in their domain."
        )
        console.print(f"[bold yellow]Using default task:[/bold yellow] {default_task}")
        role_play_kwargs["task_prompt"] = default_task
    
    # Create the role-playing session
    role_play_session = RolePlaying(**role_play_kwargs)
    
    # Print system and task messages
    console.print("\n[bold]System Messages:[/bold]")
    console.print(f"[bold blue]Assistant System Message:[/bold blue]")
    console.print(f"{role_play_session.assistant_sys_msg}\n")
    console.print(f"[bold green]User System Message:[/bold green]")
    console.print(f"{role_play_session.user_sys_msg}\n")
    console.print(f"[bold yellow]Task Prompt:[/bold yellow]")
    console.print(f"{role_play_session.task_prompt}\n")
    
    # Initialize the chat
    console.print("[bold]Starting chat...[/bold]\n")
    input_msg = role_play_session.init_chat()
    chat_history = []
    
    # Add system messages to chat history
    chat_history.append({
        "role": f"system ({assistant_role})",
        "content": role_play_session.assistant_sys_msg
    })
    chat_history.append({
        "role": f"system ({user_role})",
        "content": role_play_session.user_sys_msg
    })
    chat_history.append({
        "role": "task",
        "content": role_play_session.task_prompt
    })
    
    # Execute the conversation
    n = 0
    while n < chat_turn_limit:
        n += 1
        assistant_response, user_response = role_play_session.step(input_msg)
        
        # Check for termination
        if assistant_response.terminated:
            console.print(
                f"[bold red]Assistant terminated. "
                f"Reason: {assistant_response.info['termination_reasons']}.[/bold red]"
            )
            break
            
        if user_response.terminated:
            console.print(
                f"[bold red]User terminated. "
                f"Reason: {user_response.info['termination_reasons']}.[/bold red]"
            )
            break
        
        # Add responses to history and display
        chat_history.append({
            "role": user_role,
            "content": user_response.msg.content
        })
        console.print(f"[bold green]{user_role}:[/bold green]")
        console.print(f"{user_response.msg.content}\n")
        
        chat_history.append({
            "role": assistant_role,
            "content": assistant_response.msg.content
        })
        console.print(f"[bold blue]{assistant_role}:[/bold blue]")
        console.print(f"{assistant_response.msg.content}\n")
        
        # Check if task is done
        if "CAMEL_TASK_DONE" in user_response.msg.content:
            console.print("[bold yellow]Task completed successfully![/bold yellow]")
            break
            
        input_msg = assistant_response.msg
    
    # Save to file if specified
    if output_file:
        output_path = Path(output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, indent=2)
        console.print(f"\nConversation saved to: [bold blue]{output_path}[/bold blue]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an AI Society role-playing conversation")
    parser.add_argument(
        "--assistant",
        type=str,
        default="Python Programmer",
        help="Role of the assistant agent"
    )
    parser.add_argument(
        "--user",
        type=str,
        default="Product Manager",
        help="Role of the user agent"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task for the agents to perform"
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="ai_society",
        choices=["ai_society", "code", "misalignment"],
        help="Type of task to generate"
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
        "--custom-task",
        action="store_true",
        help="Generate a custom task"
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=15,
        help="Maximum number of conversation turns"
    )
    parser.add_argument(
        "--list-roles",
        type=str,
        choices=["all", "assistant", "user"],
        help="List available pre-defined roles"
    )
    
    args = parser.parse_args()
    
    # List roles if requested
    if args.list_roles:
        list_available_roles(args.list_roles)
        exit()
    
    main(
        assistant_role=args.assistant,
        user_role=args.user,
        task_prompt=args.task,
        task_type=args.task_type,
        oumi_model_path=args.oumi_model,
        camel_model_name=args.camel_model,
        output_file=args.output,
        custom_task=args.custom_task,
        chat_turn_limit=args.turns,
    ) 
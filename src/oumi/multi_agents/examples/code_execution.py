"""
Example showing how to use CAMEL's code execution capabilities with Oumi.

This example demonstrates how to create agents that can write and execute code.
"""

import argparse
import os
from typing import Optional

from rich.console import Console
from rich.syntax import Syntax

from oumi.multi_agents import ChatAgent
from oumi.multi_agents.adapters import create_agent_model
from oumi.multi_agents.api import InternalPythonInterpreter, CodeExecutionToolkit


console = Console()


def main(
    coding_task: str = "Write a function to calculate the first 10 Fibonacci numbers",
    oumi_model_path: Optional[str] = None,
    camel_model_name: Optional[str] = None,
    sandbox: bool = True,
):
    """Run a chat with an agent that can write and execute code.

    Args:
        coding_task: The coding task to give to the agent.
        oumi_model_path: Optional path to an Oumi model.
        camel_model_name: Optional name of a CAMEL model.
        sandbox: Whether to use a sandboxed interpreter for safety.
    """
    console.print(f"[bold green]Coding task:[/bold green] {coding_task}")
    
    # Decide which model to use
    model = None
    if oumi_model_path:
        console.print(f"[bold yellow]Using Oumi model:[/bold yellow] {oumi_model_path}")
        model = create_agent_model(model_name_or_path=oumi_model_path)
    elif camel_model_name:
        console.print(f"[bold yellow]Using CAMEL model:[/bold yellow] {camel_model_name}")
        model = camel_model_name
    
    # Create the code execution toolkit with the appropriate interpreter
    if sandbox:
        console.print("[bold blue]Using sandboxed interpreter[/bold blue]")
        # Using default interpreter which is safe
        toolkit = CodeExecutionToolkit()
    else:
        console.print("[bold red]Using internal Python interpreter (not sandboxed)[/bold red]")
        # Using internal Python interpreter which runs code in the current process
        interpreter = InternalPythonInterpreter()
        toolkit = CodeExecutionToolkit(interpreter=interpreter)
    
    # Create and use the agent with the code execution toolkit
    system_message = (
        "You are a Python programming assistant. "
        "You can write and execute Python code to solve problems. "
        "Always show your reasoning and explain your code."
    )
    agent = ChatAgent(system_message, model=model)
    agent.add_toolkit(toolkit)
    agent.reset()
    
    # Create a prompt that encourages the agent to use code execution
    prompt = (
        f"Please help me with this coding task: {coding_task}\n\n"
        "Write the code, then execute it to verify it works correctly. "
        "Explain your approach and the output."
    )
    
    console.print("[bold]Sending coding task to agent...[/bold]")
    response = agent.step(prompt)
    
    console.print("\n[bold]Response:[/bold]")
    console.print(response.msg.content)
    
    # Extract and highlight code blocks for better visualization
    content = response.msg.content
    code_blocks = []
    lines = content.split("\n")
    in_code_block = False
    current_block = []
    
    for line in lines:
        if line.strip().startswith("```python") or line.strip() == "```python":
            in_code_block = True
            current_block = []
        elif line.strip() == "```" and in_code_block:
            in_code_block = False
            if current_block:
                code_blocks.append("\n".join(current_block))
        elif in_code_block:
            current_block.append(line)
    
    if code_blocks:
        console.print("\n[bold]Generated Code Blocks:[/bold]")
        for i, code in enumerate(code_blocks, 1):
            console.print(f"\n[bold]Code Block {i}:[/bold]")
            syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
            console.print(syntax)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a chat with an agent that can write and execute code")
    parser.add_argument(
        "--task",
        type=str,
        default="Write a function to calculate the first 10 Fibonacci numbers",
        help="Coding task to give to the agent"
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
        "--no-sandbox",
        action="store_true",
        help="Disable sandbox for code execution (not recommended)"
    )
    
    args = parser.parse_args()
    main(
        coding_task=args.task,
        oumi_model_path=args.oumi_model,
        camel_model_name=args.camel_model,
        sandbox=not args.no_sandbox,
    ) 
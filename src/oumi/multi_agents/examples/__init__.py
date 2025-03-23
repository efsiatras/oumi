"""
Examples of using CAMEL-AI multi-agent capabilities with Oumi.

This package contains example scripts that demonstrate various ways to use
CAMEL-AI's multi-agent capabilities through Oumi's integration.
"""

# List of examples
EXAMPLES = {
    "single_agent": "Simple example of using a single CAMEL agent",
    "role_playing": "Role-playing conversation between two agents",
    "toolkit_usage": "Using CAMEL toolkits with agents",
    "code_execution": "Using agents that can write and execute code",
    "agent_memory": "Creating agents with persistent memory",
    "ai_society_role_playing": "Advanced AI Society role-playing scenarios",
}


def list_examples():
    """Print a list of all available examples."""
    print("\nAvailable CAMEL Examples in Oumi:")
    print("=================================")
    
    for name, description in EXAMPLES.items():
        print(f"{name}: {description}")
    
    print("\nTo run an example:")
    print("python -m oumi.multi_agents.examples.<example_name> --help")
    
    print("\nFor example:")
    print("python -m oumi.multi_agents.examples.single_agent --prompt \"Tell me about quantum computing\"")


if __name__ == "__main__":
    list_examples() 
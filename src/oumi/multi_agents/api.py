"""
Complete CAMEL-AI API surface exposed for Oumi.

This module re-exports all relevant CAMEL-AI modules and classes
to provide the full API surface within Oumi.
"""

# Re-export all CAMEL-AI components
# ================================

# Agents
from camel.agents.base import BaseAgent
from camel.agents.chat_agent import ChatAgent
from camel.agents.critic_agent import CriticAgent
from camel.agents.deductive_reasoner_agent import DeductiveReasonerAgent
from camel.agents.embodied_agent import EmbodiedAgent
from camel.agents.knowledge_graph_agent import KnowledgeGraphAgent
from camel.agents.role_assignment_agent import RoleAssignmentAgent
from camel.agents.search_agent import SearchAgent
from camel.agents.task_agent import TaskAgent

# Configs
from camel.configs.anthropic_config import AnthropicConfig
from camel.configs.base_config import BaseConfig
from camel.configs.gemini_config import GeminiConfig
from camel.configs.groq_config import GroqConfig
from camel.configs.litellm_config import LiteLLMConfig
from camel.configs.mistral_config import MistralConfig
from camel.configs.ollama_config import OllamaConfig
from camel.configs.openai_config import OpenAIConfig
from camel.configs.reka_config import RekaConfig
from camel.configs.samba_config import SambaConfig
from camel.configs.togetherai_config import TogetherAIConfig
from camel.configs.vllm_config import VLLMConfig
from camel.configs.zhipuai_config import ZhipuAIConfig

# Data Generation
from camel.datagen.cot_datagen import CoTDataGenerator
from camel.datagen.self_improving_cot import SelfImprovingCoTDataGenerator

# Embeddings
from camel.embeddings.base import BaseEmbedding
from camel.embeddings.mistral_embedding import MistralEmbedding
from camel.embeddings.openai_embedding import OpenAIEmbedding
from camel.embeddings.sentence_transformers_embeddings import SentenceTransformersEmbedding
from camel.embeddings.vlm_embedding import VLMEmbedding

# Interpreters
from camel.interpreters.base import BaseInterpreter
from camel.interpreters.docker_interpreter import DockerInterpreter
from camel.interpreters.internal_python_interpreter import InternalPythonInterpreter
from camel.interpreters.interpreter_error import InterpreterError
from camel.interpreters.ipython_interpreter import IPythonInterpreter
from camel.interpreters.subprocess_interpreter import SubprocessInterpreter

# Loaders
from camel.loaders.base_io import BaseIO
from camel.loaders.firecrawl_reader import FirecrawlReader
from camel.loaders.jina_url_reader import JinaURLReader
from camel.loaders.unstructured_io import UnstructuredIO

# Memories
from camel.memories.agent_memories import AgentMemories
from camel.memories.base import BaseMemory
from camel.memories.records import ChatRecord, FunctionCallingRecord

# Messages
from camel.messages.base import BaseMessage, SystemMessage, HumanMessage, AssistantMessage
from camel.messages.func_message import FunctionCall, FunctionMessage

# Models
from camel.models.anthropic_model import AnthropicModel
from camel.models.azure_openai_model import AzureOpenAIModel
from camel.models.base_model import BaseModelBackend
from camel.models.gemini_model import GeminiModel
from camel.models.groq_model import GroqModel
from camel.models.litellm_model import LiteLLMModel
from camel.models.mistral_model import MistralModel
from camel.models.model_factory import ModelFactory
from camel.models.nemotron_model import NemotronModel
from camel.models.ollama_model import OllamaModel
from camel.models.open_source_model import OpenSourceModel
from camel.models.openai_compatible_model import OpenAICompatibleModel
from camel.models.openai_model import OpenAIModel
from camel.models.reka_model import RekaModel
from camel.models.samba_model import SambaModel
from camel.models.stub_model import StubModel
from camel.models.togetherai_model import TogetherAIModel
from camel.models.vllm_model import VLLMModel
from camel.models.zhipuai_model import ZhipuAIModel

# Prompts
from camel.prompts.ai_society import AISocietyPromptTemplateDict, AI_SOCIETY_SYSTEM_MESSAGE
from camel.prompts.base import (
    BasePromptTemplate,
    PromptTemplateGenerator,
    SystemMessagePromptTemplate,
)
from camel.prompts.code import CODE_SYSTEM_MESSAGE
from camel.prompts.evaluation import EVALUATION_SYSTEM_MESSAGE
from camel.prompts.generate_text_embedding_data import GENERATE_TEXT_EMBEDDING_DATA_SYSTEM_MESSAGE
from camel.prompts.image_craft import (
    IMAGE_CRAFT_SYSTEM_MESSAGE_FOR_DALLE,
    IMAGE_CRAFT_SYSTEM_MESSAGE_FOR_HUMAN,
    IMAGE_CRAFT_SYSTEM_MESSAGE_FOR_STABLE_DIFFUSION,
)
from camel.prompts.misalignment import MISALIGNMENT_SCENARIO_SYSTEM_MESSAGE
from camel.prompts.multi_condition_image_craft import MULTI_CONDITION_IMAGE_CRAFT_SYSTEM_MESSAGE
from camel.prompts.object_recognition import OBJECT_RECOGNITION_SYSTEM_MESSAGE
from camel.prompts.prompt_templates import DEFAULT_ROLE_PROMPT_TEMPLATE
from camel.prompts.role_description_prompt_template import RoleDescriptionPromptTemplate
from camel.prompts.solution_extraction import SOLUTION_EXTRACTION_SYSTEM_MESSAGE
from camel.prompts.task_prompt_template import TaskPromptTemplate
from camel.prompts.translation import TRANSLATION_SYSTEM_MESSAGE
from camel.prompts.video_description_prompt import VIDEO_DESCRIPTION_SYSTEM_MESSAGE

# Responses
from camel.responses.agent_responses import AgentResponse

# Retrievers
from camel.retrievers.auto_retriever import AutoRetriever
from camel.retrievers.base import BaseRetriever
from camel.retrievers.bm25_retriever import BM25Retriever
from camel.retrievers.cohere_rerank_retriever import CohereRerankRetriever
from camel.retrievers.vector_retriever import VectorRetriever

# Societies
from camel.societies.babyagi_playing import BabyAGIPlaying
from camel.societies.role_playing import RolePlaying

# Tasks
from camel.tasks.task import Task
from camel.tasks.task_prompt import TaskPrompt

# Terminators
from camel.terminators.base import BaseTerminator
from camel.terminators.response_terminator import ResponseTerminator
from camel.terminators.token_limit_terminator import TokenLimitTerminator

# Toolkits (already exposed through the toolkits.py module)
from camel.toolkits.base import BaseToolkit
from camel.toolkits.code_execution import CodeExecutionToolkit
from camel.toolkits.dalle_toolkit import DalleToolkit
from camel.toolkits.github_toolkit import GitHubToolkit
from camel.toolkits.google_maps_toolkit import GoogleMapsToolkit
from camel.toolkits.linkedin_toolkit import LinkedInToolkit
from camel.toolkits.math_toolkit import MathToolkit
from camel.toolkits.open_api_toolkit import OpenAPIToolkit
from camel.toolkits.openai_function import OpenAIFunctionToolkit
from camel.toolkits.reddit_toolkit import RedditToolkit
from camel.toolkits.retrieval_toolkit import RetrievalToolkit
from camel.toolkits.search_toolkit import SearchToolkit
from camel.toolkits.slack_toolkit import SlackToolkit
from camel.toolkits.twitter_toolkit import TwitterToolkit
from camel.toolkits.weather_toolkit import WeatherToolkit

# Types
from camel.types.enums import (
    ModelType,
    RoleType,
    TaskType,
    SeparatorStyle,
    TaskPhaseCompletion,
)
from camel.types.openai_types import ModelMessage

# Utils
from camel.utils.async_func import call_async_func
from camel.utils.commons import get_random_name, openai_api_env_vars_set
from camel.utils.constants import GENERAL_ASSISTANT_ROLES, GENERAL_USER_ROLES
from camel.utils.token_counting import count_message_tokens, count_tokens 
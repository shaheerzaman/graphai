import inspect
import os
from typing import Any, Callable
from pydantic import BaseModel, Field
import logging
import sys


class ColoredFormatter(logging.Formatter):
    """Custom colored formatter for the logger using ANSI escape codes."""

    # ANSI escape codes for colors
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"

    def __init__(self):
        super().__init__(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record):
        # Check if the output supports color (TTY)
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
                record.msg = f"{self.COLORS[levelname]}{record.msg}{self.RESET}"
        return super().format(record)


def add_coloured_handler(logger):
    """Add a coloured handler to the logger."""
    formatter = ColoredFormatter()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def setup_custom_logger(name):
    """Setup a custom logger."""
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        add_coloured_handler(logger)
        
        # Set log level from environment variable, default to INFO
        log_level = os.getenv("GRAPHAI_LOG_LEVEL", "INFO").upper()
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        logger.setLevel(level_map.get(log_level, logging.INFO))
        logger.propagate = False

    return logger


logger: logging.Logger = setup_custom_logger(__name__)


def openai_type_mapping(param_type: str) -> str:
    if param_type == "int":
        return "number"
    elif param_type == "float":
        return "number"
    elif param_type == "str":
        return "string"
    elif param_type == "bool":
        return "boolean"
    else:
        return "object"


class Parameter(BaseModel):
    """Parameter for a function.

    :param name: The name of the parameter.
    :type name: str
    :param description: The description of the parameter.
    :type description: str | None
    :param type: The type of the parameter.
    :type type: str
    :param default: The default value of the parameter.
    :type default: Any
    :param required: Whether the parameter is required.
    :type required: bool
    """

    name: str = Field(description="The name of the parameter")
    description: str | None = Field(
        default=None, description="The description of the parameter"
    )
    type: str = Field(description="The type of the parameter")
    default: Any = Field(description="The default value of the parameter")
    required: bool = Field(description="Whether the parameter is required")

    def to_dict(self) -> dict[str, Any]:
        """Convert the parameter to a dictionary for an standard dictionary-based function schema.
        This is the most common format used by LLM providers, including OpenAI, Ollama, and others.

        :return: The parameter in dictionary format.
        :rtype: dict[str, Any]
        """
        return {
            self.name: {
                "description": self.description,
                "type": openai_type_mapping(self.type),
            }
        }


class FunctionSchema(BaseModel):
    """Class that consumes a function and can return a schema required by
    different LLMs for function calling.
    """

    name: str = Field(description="The name of the function")
    description: str = Field(description="The description of the function")
    signature: str = Field(description="The signature of the function")
    output: str = Field(description="The output of the function")
    parameters: list[Parameter] = Field(description="The parameters of the function")

    @classmethod
    def from_callable(cls, function: Callable) -> "FunctionSchema":
        """Initialize the FunctionSchema.

        :param function: The function to consume.
        :type function: Callable
        """
        if not callable(function):
            raise TypeError("Function must be a Callable")
        
        name = function.__name__
        doc = inspect.getdoc(function)
        description = str(doc) if doc else ""
        if not description:
            logger.warning(f"Function {name} has no docstring")
        signature = str(inspect.signature(function))
        output = str(inspect.signature(function).return_annotation)
        parameters = []
        for param in inspect.signature(function).parameters.values():
            parameters.append(
                Parameter(
                    name=param.name,
                    type=param.annotation.__name__,
                    default=param.default,
                    required=param.default is inspect.Parameter.empty,
                )
            )
        return cls.model_construct(
            name=name,
            description=description,
            signature=signature,
            output=output,
            parameters=parameters,
        )

    @classmethod
    def from_pydantic(cls, model: BaseModel) -> "FunctionSchema":
        signature_parts = []
        for field_name, field_model in model.__annotations__.items():
            field_info = model.model_fields[field_name]
            default_value = field_info.default
            if default_value:
                default_repr = repr(default_value)
                signature_part = (
                    f"{field_name}: {field_model.__name__} = {default_repr}"
                )
            else:
                signature_part = f"{field_name}: {field_model.__name__}"
            signature_parts.append(signature_part)
        signature = f"({', '.join(signature_parts)}) -> str"
        return cls.model_construct(
            name=model.__class__.__name__,
            description=model.__doc__ or "",
            signature=signature,
            output="",  # TODO: Implement output
            parameters=[],
        )

    def to_dict(self) -> dict:
        schema_dict = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        k: v
                        for param in self.parameters
                        for k, v in param.to_dict().items()
                    },
                    "required": [
                        param.name for param in self.parameters if param.required
                    ],
                },
            },
        }
        return schema_dict

    def to_openai(self) -> dict:
        return self.to_dict()


DEFAULT = set(["default", "openai", "ollama", "litellm"])


def get_schemas(callables: list[Callable], format: str = "default") -> list[dict]:
    if format in DEFAULT:
        return [
            FunctionSchema.from_callable(callable).to_dict() for callable in callables
        ]
    else:
        raise ValueError(f"Format {format} not supported")

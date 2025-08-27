from enum import Enum
import inspect
import os
import sys
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Sequence,
    Set,
    Tuple,
    Union,
    get_args,
    get_origin,
)
from pydantic import BaseModel, Field
from pydantic_core import PydanticUndefined
import logging


# we support python 3.10 so we define our own StrEnum (introduced in 3.11)
class StrEnum(str, Enum):
    """Backport of StrEnum for Python < 3.11"""

    def __str__(self):
        return self.value


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
    """Map a friendly type string to an OpenAI-compatible JSON Schema type.

    The input is a relaxed, human-friendly type string such as:
    - "int", "float", "str", "bool"
    - "list[int]", "tuple[str, int]", "set[str]"
    - "dict[str, int]", "mapping[str, any]"
    - "Optional[str]", "Union[int, str]"
    - "any", "object"
    """
    s = (param_type or "").lower()

    # Unwrap Optional[...] to its inner type
    if s.startswith("optional[") and s.endswith("]"):
        inner = s[len("optional[") : -1]
        return openai_type_mapping(inner)

    # Union[...] is ambiguous in JSON schema for OpenAI tools; degrade to object
    if s.startswith("union["):
        return "object"

    if s in {"int", "float", "number"}:
        return "number"
    if s in {"str", "string"}:
        return "string"
    if s in {"bool", "boolean"}:
        return "boolean"

    # Collections → array/object
    if (
        s.startswith("list")
        or s.startswith("tuple")
        or s.startswith("set")
        or s.startswith("sequence")
        or s.startswith("array")
    ):
        return "array"
    if s.startswith("dict") or s.startswith("mapping"):
        return "object"

    # Fallbacks
    if s in {"any", "none", "null", "object"}:
        return "object"
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


class OpenAIAPI(StrEnum):
    COMPLETIONS = "completions"
    RESPONSES = "responses"


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

        def _normalize_type_name(tp: Any) -> str:
            """Create a friendly, normalized type name string from an annotation.

            Handles typing generics (List/Dict/Union/Optional/etc.), builtins,
            and missing annotations gracefully.
            """
            if tp is inspect._empty:
                return "any"

            if tp is Any:
                return "any"

            origin = get_origin(tp)
            args = get_args(tp)

            # Union and Optional
            if origin is Union:
                non_none = [a for a in args if a is not type(None)]  # noqa: E721
                if len(non_none) == 1 and len(args) == 2:
                    return f"Optional[{_normalize_type_name(non_none[0])}]"
                inner = ", ".join(_normalize_type_name(a) for a in non_none)
                return f"Union[{inner}]"

            # Sequences
            if origin in (list,):
                inner = (
                    ", ".join(_normalize_type_name(a) for a in args) if args else "any"
                )
                return f"list[{inner}]" if args else "list"

            if origin in (List, Sequence, Set, tuple, Tuple, set):  # noqa: F821
                container = (
                    "list"
                    if origin in (List, Sequence, Set)
                    else ("tuple" if origin in (tuple, Tuple) else "set")
                )
                if container == "tuple" and len(args) == 2 and args[1] is Ellipsis:
                    # Tuple[T, ...]
                    inner = f"{_normalize_type_name(args[0])}, ..."
                else:
                    inner = (
                        ", ".join(_normalize_type_name(a) for a in args)
                        if args
                        else "any"
                    )
                return f"{container}[{inner}]" if args else container

            # Mappings / Dicts
            if origin in (dict, Dict, Mapping):
                if args:
                    key = _normalize_type_name(args[0]) if len(args) > 0 else "any"
                    val = _normalize_type_name(args[1]) if len(args) > 1 else "any"
                    return f"dict[{key}, {val}]"
                return "dict"

            # Plain classes / builtins
            try:
                name_attr = getattr(tp, "__name__", None)
                if isinstance(name_attr, str):
                    return name_attr
            except Exception:
                pass
            # Fallback to string representation
            return str(tp)

        parameters = []
        for param in inspect.signature(function).parameters.values():
            type_name = _normalize_type_name(param.annotation)
            parameters.append(
                Parameter(
                    name=param.name,
                    type=type_name,
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
    def from_pydantic(cls, model: type[BaseModel]) -> "FunctionSchema":
        """Create a FunctionSchema from a Pydantic model class.

        :param model: The Pydantic model class to convert
        :type model: type[BaseModel]
        :return: FunctionSchema instance
        :rtype: FunctionSchema
        """
        # Extract model metadata
        name = model.__name__
        description = model.__doc__ or ""

        # Build parameters list
        parameters = []
        signature_parts = []

        for field_name, field_info in model.model_fields.items():
            # Get the field type
            field_type = model.__annotations__.get(field_name)

            # Determine the type name - handle Optional and other generic types
            type_name = str(field_type)

            # Try to extract the actual type from Optional[T] -> T
            origin = get_origin(field_type)
            args = get_args(field_type)

            if origin is Union:
                # This is likely Optional[T] which is Union[T, None]
                non_none_types = [arg for arg in args if arg is not type(None)]
                if non_none_types:
                    actual_type = non_none_types[0]
                    if hasattr(actual_type, "__name__"):
                        type_name = actual_type.__name__
                    else:
                        type_name = str(actual_type)
            elif field_type and hasattr(field_type, "__name__"):
                type_name = field_type.__name__

            # Check if field is required (no default value)
            # In Pydantic v2, PydanticUndefined means no default
            is_required = (
                field_info.default is PydanticUndefined
                and field_info.default_factory is None
            )

            # Get the actual default value
            if (
                field_info.default is not PydanticUndefined
                and field_info.default is not None
            ):
                default_value = field_info.default
            elif field_info.default_factory is not None:
                # For default_factory, we can't always call it without arguments
                # Just use a placeholder to indicate there's a factory
                try:
                    # Try calling with no arguments (common case)
                    default_value = field_info.default_factory()  # type: ignore[call-arg]
                except TypeError:
                    # If it needs arguments, just indicate it has a factory default
                    default_value = "<factory>"
            else:
                default_value = inspect.Parameter.empty

            # Add parameter
            parameters.append(
                Parameter(
                    name=field_name,
                    description=field_info.description,
                    type=type_name,
                    default=default_value,
                    required=is_required,
                )
            )

            # Build signature part
            if default_value != inspect.Parameter.empty:
                signature_parts.append(
                    f"{field_name}: {type_name} = {repr(default_value)}"
                )
            else:
                signature_parts.append(f"{field_name}: {type_name}")

        signature = f"({', '.join(signature_parts)}) -> dict"

        return cls.model_construct(
            name=name,
            description=description,
            signature=signature,
            output="dict",
            parameters=parameters,
        )

    def to_dict(self) -> dict[str, Any]:
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

    def to_openai(self, api: OpenAIAPI = OpenAIAPI.COMPLETIONS) -> dict[str, Any]:
        """Convert the function schema into OpenAI-compatible formats. Supports
        both completions and responses APIs.

        :param api: The API to convert to.
        :type api: OpenAIAPI
        :return: The function schema in OpenAI-compatible format.
        :rtype: dict
        """
        if api == "completions":
            return self.to_dict()
        elif api == "responses":
            return {
                "type": "function",
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
            }
        else:
            raise ValueError(f"Unrecognized OpenAI API: {api}")


DEFAULT = set(["default", "openai", "ollama", "litellm"])


def get_schemas(
    callables: list[Callable], format: str = "default"
) -> list[dict[str, Any]]:
    if format in DEFAULT:
        return [
            FunctionSchema.from_callable(callable).to_dict() for callable in callables
        ]
    else:
        raise ValueError(f"Format {format} not supported")

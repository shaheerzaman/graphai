import inspect
from typing import Optional, Any, Union, List, Dict, Tuple, Set

from pydantic import BaseModel, Field

from graphai.utils import FunctionSchema


def scrape_webpage(url: str, name: str = "test") -> str:
    """Provides access to web scraping. You can use this tool to scrape a webpage.
    Many webpages may return no information due to JS or adblock issues, if this
    happens, you must use a different URL.
    """
    return "hello there"


def test_function_schema():
    schema = FunctionSchema.from_callable(scrape_webpage)
    assert schema.name == scrape_webpage.__name__
    assert schema.description == str(inspect.getdoc(scrape_webpage))
    assert schema.signature == str(inspect.signature(scrape_webpage))
    assert schema.output == str(inspect.signature(scrape_webpage).return_annotation)
    assert len(schema.parameters) == 2


def test_function_schema_to_dict():
    schema = FunctionSchema.from_callable(scrape_webpage)
    dict_schema = schema.to_dict()
    assert dict_schema["type"] == "function"
    assert dict_schema["function"]["name"] == schema.name
    assert dict_schema["function"]["description"] == schema.description
    assert dict_schema["function"]["parameters"]["type"] == "object"
    assert (
        dict_schema["function"]["parameters"]["properties"]["url"]["type"] == "string"
    )
    assert (
        dict_schema["function"]["parameters"]["properties"]["name"]["type"] == "string"
    )
    assert (
        dict_schema["function"]["parameters"]["properties"]["url"]["description"]
        is None
    )
    assert (
        dict_schema["function"]["parameters"]["properties"]["name"]["description"]
        is None
    )
    assert dict_schema["function"]["parameters"]["required"] == ["url"]


def test_function_schema_to_openai_responses():
    schema = FunctionSchema.from_callable(scrape_webpage)
    openai_responses_schema = schema.to_openai(api="responses")  # type: ignore

    # Check that the structure is different from to_dict()
    assert openai_responses_schema["type"] == "function"
    assert openai_responses_schema["name"] == schema.name
    assert openai_responses_schema["description"] == schema.description

    # Check parameters structure
    assert openai_responses_schema["parameters"]["type"] == "object"
    assert (
        openai_responses_schema["parameters"]["properties"]["url"]["type"] == "string"
    )
    assert (
        openai_responses_schema["parameters"]["properties"]["name"]["type"] == "string"
    )
    assert openai_responses_schema["parameters"]["required"] == ["url"]

    # Verify it doesn't have the nested "function" key like to_dict()
    assert "function" not in openai_responses_schema


class WebSearchArgs(BaseModel):
    """Arguments for web search functionality."""

    query: str = Field(description="The search query to execute")
    max_results: int = Field(
        default=10, description="Maximum number of results to return"
    )
    include_snippets: Optional[bool] = Field(
        default=True, description="Whether to include text snippets"
    )


def test_function_schema_from_pydantic():
    schema = FunctionSchema.from_pydantic(WebSearchArgs)

    # Check basic schema properties
    assert schema.name == "WebSearchArgs"
    assert schema.description == "Arguments for web search functionality."

    # Check parameters were extracted correctly
    assert len(schema.parameters) == 3

    # Find each parameter by name
    params_by_name = {p.name: p for p in schema.parameters}

    # Check query parameter
    assert "query" in params_by_name
    query_param = params_by_name["query"]
    assert query_param.type == "str"
    assert query_param.required is True
    assert query_param.description == "The search query to execute"

    # Check max_results parameter
    assert "max_results" in params_by_name
    max_results_param = params_by_name["max_results"]
    assert max_results_param.type == "int"
    assert max_results_param.required is False
    assert max_results_param.default == 10
    assert max_results_param.description == "Maximum number of results to return"

    # Check include_snippets parameter
    assert "include_snippets" in params_by_name
    snippets_param = params_by_name["include_snippets"]
    assert snippets_param.type == "bool"
    assert snippets_param.required is False
    assert snippets_param.default is True
    assert snippets_param.description == "Whether to include text snippets"


def test_function_schema_from_pydantic_to_dict():
    schema = FunctionSchema.from_pydantic(WebSearchArgs)
    dict_schema = schema.to_dict()

    # Check structure
    assert dict_schema["type"] == "function"
    assert dict_schema["function"]["name"] == "WebSearchArgs"
    assert (
        dict_schema["function"]["description"]
        == "Arguments for web search functionality."
    )

    # Check parameters structure
    props = dict_schema["function"]["parameters"]["properties"]
    assert "query" in props
    assert props["query"]["type"] == "string"
    assert props["query"]["description"] == "The search query to execute"

    assert "max_results" in props
    assert props["max_results"]["type"] == "number"
    assert props["max_results"]["description"] == "Maximum number of results to return"

    assert "include_snippets" in props
    assert props["include_snippets"]["type"] == "boolean"
    assert (
        props["include_snippets"]["description"] == "Whether to include text snippets"
    )

    # Check required fields
    assert dict_schema["function"]["parameters"]["required"] == ["query"]


def test_function_schema_from_callable_generics_and_unannotated():
    """Ensure from_callable handles generics, unions/optionals, and unannotated params."""

    def tool(
        a: List[int],
        b: Dict[str, float],
        c: Optional[str] = None,
        d: Union[int, str] = 0,
        e=None,
        f: Any = None,
        g: Tuple[int, int] = (1, 2),
        h: Set[str] = set(),
    ) -> str:
        """A sample tool with various parameter types."""
        return "ok"

    schema = FunctionSchema.from_callable(tool)
    dict_schema = schema.to_dict()
    props = dict_schema["function"]["parameters"]["properties"]

    # Collections map to array/object
    assert props["a"]["type"] == "array"  # List[int]
    assert props["b"]["type"] == "object"  # Dict[str, float]
    assert props["g"]["type"] == "array"  # Tuple[int, int]
    assert props["h"]["type"] == "array"  # Set[str]

    # Optionals/Unions
    assert props["c"]["type"] == "string"  # Optional[str] -> string
    assert props["d"]["type"] == "object"  # Union[int, str] -> object (fallback)

    # Unannotated / Any
    assert props["e"]["type"] == "object"
    assert props["f"]["type"] == "object"

    # Required detection (only a, b have no default)
    required = set(dict_schema["function"]["parameters"]["required"])
    assert required == {"a", "b"}


def test_function_schema_from_callable_builtin_generics():
    """Support builtin generic syntax like list[int] and dict[str, int]."""

    def tool2(items: list[int], meta: dict[str, int] = {}):  # noqa: B006 - fine for test
        return "ok"

    schema = FunctionSchema.from_callable(tool2)
    dict_schema = schema.to_dict()
    props = dict_schema["function"]["parameters"]["properties"]

    assert props["items"]["type"] == "array"
    assert props["meta"]["type"] == "object"
    assert dict_schema["function"]["parameters"]["required"] == ["items"]

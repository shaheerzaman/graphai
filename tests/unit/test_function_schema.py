import inspect

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
    openai_responses_schema = schema.to_openai(api="responses")
    
    # Check that the structure is different from to_dict()
    assert openai_responses_schema["type"] == "function"
    assert openai_responses_schema["name"] == schema.name
    assert openai_responses_schema["description"] == schema.description
    
    # Check parameters structure
    assert openai_responses_schema["parameters"]["type"] == "object"
    assert openai_responses_schema["parameters"]["properties"]["url"]["type"] == "string"
    assert openai_responses_schema["parameters"]["properties"]["name"]["type"] == "string"
    assert openai_responses_schema["parameters"]["required"] == ["url"]
    
    # Verify it doesn't have the nested "function" key like to_dict()
    assert "function" not in openai_responses_schema

from typing import Any, Optional, Literal, Union, Dict, Type
from pydantic import BaseModel
from langchain_core.language_models import BaseLanguageModel
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_community.chat_models import ChatOllama
from langchain_core.utils.function_calling import convert_to_openai_tool

class StructuredLanguageModel(ChatOllama):



    def with_structured_output(
        self,
        schema: Optional[Union[Dict, Type[BaseModel]]] = None,
        *,
        method: Literal["function_calling", "json_mode"] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ):

        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = isinstance(schema, type) and issubclass(schema, BaseModel)
        if method == "function_calling":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is 'function_calling'. "
                    "Received None."
                )
            # Assuming bind_tools and other necessary functions are implemented
            #llm = self.bind_tools([schema], tool_choice=True)
            if is_pydantic_schema:
                output_parser = PydanticToolsParser(
                    tools=[schema], first_tool_only=True
                )
            else:
                key_name = convert_to_openai_tool(schema)["function"]["name"]
                output_parser = JsonOutputKeyToolsParser(
                    key_name=key_name, first_tool_only=True
                )

        elif method == "json_mode":
            llm = self.bind(response_format={"type": "json_object"})
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)
                if is_pydantic_schema
                else JsonOutputParser()
            )
        else:
            raise ValueError(
                f"Unrecognized method argument. Expected one of 'function_calling' or "
                f"'json_format'. Received: '{method}'"
            )

        return self | output_parser


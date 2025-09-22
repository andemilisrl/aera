# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from random import choices
from string import ascii_letters, digits
from typing import Union, Optional

import partial_json_parser
import regex as re
from partial_json_parser.core.options import Allow
from pydantic import Field

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.entrypoints.openai.tool_parsers.utils import (
    extract_intermediate_diff)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)

ALPHANUMERIC = ascii_letters + digits


class CustomToolCall(ToolCall):
    id: str = Field(
        default_factory=lambda: CustomToolCall.generate_random_id())

    @staticmethod
    def generate_random_id():
        # Generate a random alphanumeric ID with 'call_' prefix to match expected format
        return "call_" + "".join(choices(ALPHANUMERIC, k=24))

    @staticmethod
    def is_valid_id(id: str) -> bool:
        return id.isalnum() and len(id) >= 1


@ToolParserManager.register_module("toolcall")
class CustomToolParser(ToolParser):
    """
    Tool call parser for custom <tool_call> format.
    
    Format:
    <tool_call>
    id: tool_id
    name: function_name
    arguments: {"key": "value"}
    </tool_call>

    Used when --enable-auto-tool-choice --tool-call-parser custom are all set
    """

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        # initialize properties used for state when parsing tool calls in
        # streaming mode
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []
        
        # Tool call parsing patterns
        self.start_tag = "<tool_call>"
        self.end_tag = "</tool_call>"
        
        # Complete tool call regex
        self.tool_call_regex = re.compile(
            r"<tool_call>\s*"
            r"id:\s*(?P<id>[^\n]+)\s*"
            r"name:\s*(?P<name>[^\n]+)\s*"
            r"arguments:\s*(?P<args>\{.*?\})\s*"
            r"</tool_call>",
            re.DOTALL
        )
        
        # Streaming parsing state
        self.parsing_state = "BUFFERING"  # BUFFERING, NORMAL, IN_TOOL_CALL
        self.current_tool_data = {}
        self.arguments_buffer = ""
        self.sent_initial_message = False
        self.initial_buffer = ""  # Buffer to hold initial content

    def adjust_request(
            self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        # No special token handling needed for our custom format
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Extract the tool calls from a complete model response.
        """

        # Check if there are any tool calls in the output
        if self.start_tag not in model_output:
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

        try:
            # Find all tool calls using regex
            matches = list(self.tool_call_regex.finditer(model_output))
            
            if not matches:
                # No valid tool calls found, return as content
                return ExtractedToolCallInformation(tools_called=False,
                                                    tool_calls=[],
                                                    content=model_output)

            tool_calls: list[CustomToolCall] = []
            
            for match in matches:
                tool_id = match.group('id').strip()
                function_name = match.group('name').strip()
                arguments_str = match.group('args').strip()
                
                # Parse arguments JSON
                try:
                    arguments_dict = json.loads(arguments_str)
                    arguments_json = json.dumps(arguments_dict, ensure_ascii=False)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse arguments JSON: {arguments_str}")
                    arguments_json = arguments_str

                tool_call = CustomToolCall(
                    id=tool_id,
                    type="function",
                    function=FunctionCall(
                        name=function_name,
                        arguments=arguments_json
                    )
                )
                tool_calls.append(tool_call)

            # Extract content before the first tool call
            first_tool_start = model_output.find(self.start_tag)
            content = model_output[:first_tool_start].strip() if first_tool_start > 0 else None
            
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:

        # Initial buffering state - we need to determine if we start with content or tool call
        if self.parsing_state == "BUFFERING":
            # Buffer until we can determine what we're starting with
            self.initial_buffer = current_text
            
            # Check if we have enough to determine the structure
            if self.start_tag in current_text:
                # We found a tool call
                tool_start_pos = current_text.find(self.start_tag)
                
                if tool_start_pos == 0:
                    # Tool call starts immediately - transition directly to tool call parsing
                    self.parsing_state = "IN_TOOL_CALL"
                    self.current_tool_id += 1
                    self.current_tool_name_sent = False
                    self.streamed_args_for_tool.append("")
                    self.current_tool_data = {}
                    self.arguments_buffer = ""
                    
                    # Process the tool call content
                    tool_content = current_text[len(self.start_tag):]
                    return self._parse_tool_call_streaming(tool_content, current_text)
                else:
                    # There's content before the tool call
                    self.parsing_state = "NORMAL"
                    content_before = current_text[:tool_start_pos]
                    # Return initial message with content and role
                    delta = DeltaMessage(content=content_before, role="assistant")
                    self.sent_initial_message = True
                    return delta
            elif len(current_text) > 20:  # Reasonable buffer size
                # No tool call found yet, assume we're starting with content
                self.parsing_state = "NORMAL"
                delta = DeltaMessage(content=current_text, role="assistant")
                self.sent_initial_message = True
                return delta
            else:
                # Keep buffering
                return None

        # Normal content streaming state
        if self.parsing_state == "NORMAL":
            # Check if tool call is starting
            if self.start_tag in current_text and self.start_tag not in previous_text:
                # Initialize for tool call parsing
                self.parsing_state = "IN_TOOL_CALL"
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                self.current_tool_data = {}
                self.arguments_buffer = ""
                
                # Don't return anything here, let the tool call parsing handle it
                return None
            
            # No tool call yet, stream content normally
            if self.start_tag not in current_text:
                return DeltaMessage(content=delta_text)

        # We're in a tool call - parse it
        if self.parsing_state == "IN_TOOL_CALL":
            tool_start = current_text.find(self.start_tag)
            if tool_start == -1:
                return None
            
            # Extract everything after <tool_call>
            tool_content = current_text[tool_start + len(self.start_tag):]
            
            # Try to parse the tool call
            return self._parse_tool_call_streaming(tool_content, current_text)

        return None

    def _parse_tool_call_streaming(self, tool_content: str, current_text: str) -> Union[DeltaMessage, None]:
        """
        Parse tool call content progressively for streaming.
        """
        
        # Extract ID if we haven't yet
        if 'id' not in self.current_tool_data:
            # Wait until we see the next field to ensure complete ID
            if '\nname:' in tool_content or self.end_tag in tool_content:
                id_match = re.search(r'id:\s*([^\n]+)', tool_content)
                if id_match:
                    self.current_tool_data['id'] = id_match.group(1).strip()

        # Extract name if we haven't yet and have ID
        if 'name' not in self.current_tool_data and 'id' in self.current_tool_data:
            # Wait until we see arguments to ensure complete name
            if '\narguments:' in tool_content or self.end_tag in tool_content:
                # Extract everything between name: and arguments: (or end)
                name_match = re.search(r'name:\s*(.*?)(?=\narguments:|$)', tool_content, re.DOTALL)
                if name_match:
                    # Clean up the name
                    name = name_match.group(1).strip().replace('\n', ' ').strip()
                    self.current_tool_data['name'] = name

        # Send initial tool call delta with ID and name
        if (not self.current_tool_name_sent and 
            'id' in self.current_tool_data and 
            'name' in self.current_tool_data):
            
            self.current_tool_name_sent = True
            
            # Generate proper tool ID
            tool_id = self.current_tool_data['id']
            if not tool_id.startswith('call_'):
                tool_id = CustomToolCall.generate_random_id()
            
            # First tool call delta includes role only if it's the very first message
            delta = DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        type="function",
                        id=tool_id,
                        function=DeltaFunctionCall(
                            name=self.current_tool_data['name'],
                            arguments=""
                        ).model_dump(exclude_none=True)
                    )
                ]
            )
            
            # Add role only on the very first message
            if not self.sent_initial_message:
                delta.role = "assistant"
                self.sent_initial_message = True
                
            return delta

        # Stream arguments character by character
        if self.current_tool_name_sent:
            # Find where the JSON arguments start
            args_match = re.search(r'arguments:\s*(\{)', tool_content)
            if args_match:
                # Get position of the opening brace
                json_start = args_match.start(1)
                args_json = tool_content[json_start:]
                
                # Find where JSON ends (look for closing brace before end tag)
                end_match = re.search(r'\}(?=\s*\n*</tool_call>)', args_json)
                if end_match:
                    args_json = args_json[:end_match.end()]
                
                # Stream only new characters
                if len(args_json) > len(self.arguments_buffer):
                    new_chars = args_json[len(self.arguments_buffer):]
                    self.arguments_buffer = args_json
                    
                    if new_chars:
                        return DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(
                                        arguments=new_chars
                                    ).model_dump(exclude_none=True)
                                )
                            ]
                        )
                
                # Check if tool call is complete
                if self.end_tag in tool_content and end_match:
                    self.parsing_state = "NORMAL"
                    # Send empty delta to signal completion
                    return DeltaMessage(content=None)

        return None
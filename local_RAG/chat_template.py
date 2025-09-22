import json
import jinja2
import random

import json, re
from typing import List, Dict, Any, Tuple

TOOL_RE       = re.compile(r"<tool>\s*(.*?)\s*</tool>", re.DOTALL)
IM_BLOCK_RE   = re.compile(r"<\|im_start\|\>(.*?)<\|im_end\|\>", re.DOTALL)
TOOLCALL_RE   = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
KV_RE         = re.compile(r"^\s*([^:]+?):\s*(.*)$")

def format_with_chatml(
    messages, tools=[], add_generation_prompt=False, template="chatml"
):
    """
    Format messages using the ChatML template without requiring a tokenizer.

    Args:
        messages: List of message dictionaries with role and content keys
        tools: Optional list of tool definitions
        add_generation_prompt: Whether to add the generation prompt for assistant

    Returns:
        Formatted chat string in ChatML format
    """
    system = messages[0]["content"] if messages[0]["role"] == "system" else ""
    messages = messages[1:] if system else messages
    system = f"{system}" if system else ""
    dnl = "\n\n" if system else ""
    system += f"{dnl}Available tools:" if tools else ""

    if not system and len(tools) == 0:
        if random.random() > 0.2:
            system = ""

    # Define the improved ChatML template
    if template == "chatml":
        chatml_template = """

{%- if system %}   
{{- '<|im_start|>system\n' + system + '\n' }}
{%- endif %}
{%- for tool in tools %}
{{- '<tool>\n' }}
{{- 'name: ' + tool['function']['name'] + '\n' }}
{{- 'description: ' + tool['function']['description'] + '\n' }}
{{- 'parameters:\n' }}
{%- for param_name, param_info in tool['function']['parameters']['properties'].items() %}
{{- '  ' + param_name + ': ' + param_info['description'] + '\n' }}
{%- endfor %}
{%- if tool['function']['parameters'].get('required') %}
{{- 'required: ' + tool['function']['parameters']['required']|tojson + '\n' }}
{%- endif %}
{{- '</tool>\n' }}
{%- endfor %}
{%- if system or tools %}
{{- '<|im_end|>\n' }}
{%- endif %}

{%- for message in messages %}
    {%- if message['role'] == 'system' %}
        {{- '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}
    {%- elif message['role'] == 'user' %}
        {{- '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}
    {%- elif message['role'] == 'assistant' and message.get('tool_calls') %}
        {{- '<|im_start|>assistant\n' }}
        {%- for tool_call in message['tool_calls'] %}
            {{- '<tool_call>\n' }}
            {{- 'id: ' + tool_call['id'] + '\n' }}
            {{- 'name: ' + tool_call['function']['name'] + '\n' }}
            {{- 'arguments: ' + tool_call['function']['arguments'] + '\n' }}
            {{- '</tool_call>\n' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}
    {%- elif message['role'] == 'tool' %}
        {{- '<|im_start|>tool\n' }}
        {{- 'tool_call_id: ' + message.get('name', message['tool_call_id']) + '\n' }}
        {{- 'content: ' + message['content'] + '\n' }}
        {{- '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"""
    else:  # use gemma template
        chatml_template = """"""

    # Create Jinja environment and template
    env = jinja2.Environment()
    template = env.from_string(chatml_template)

    # Render the template with messages and other variables
    formatted_chat = template.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        system=system,
    )

    return formatted_chat



def _kv_lines(blob: str) -> Dict[str, str]:
    "Turn indented key: value lines into a dict."
    out = {}
    for ln in blob.splitlines():
        m = KV_RE.match(ln)
        if m:
            out[m[1].strip()] = m[2].strip()
    return out

def _parse_tool_block(blob: str) -> Dict[str, Any]:
    meta = _kv_lines(blob)
    # reconstruct the OpenAI-style wrapper
    return {
        "type": "function",
        "function": {
            "name": meta.pop("name"),
            "description": meta.pop("description", ""),
            "parameters": _maybe_json(meta.pop("parameters", "{}")),
            **meta,  # holds 'required:' or any extras
        },
    }

def _parse_tool_call(blob: str) -> Dict[str, Any]:
    meta = _kv_lines(blob)
    return {
        "id": meta["id"],
        "type": "function",
        "function": {
            "name": meta["name"],
            "arguments": meta["arguments"],
        },
    }

def _maybe_json(txt: str):
    try:
        return json.loads(txt)
    except Exception:
        return txt

# ---------------------------------------------------------------------------

def parse_chatml(chat: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Reverse of `format_with_chatml`:  ChatML ➜ (messages, tools)
    """
    # 1️⃣  carve out <tool> … </tool> blocks first
    tool_defs = [ _parse_tool_block(m.group(1)) for m in TOOL_RE.finditer(chat) ]
    chat_wo_tools = TOOL_RE.sub("", chat)

    # 2️⃣  iterate over every <|im_start|> … <|im_end|> block
    messages: List[Dict[str, Any]] = []
    for block in IM_BLOCK_RE.finditer(chat_wo_tools):
        payload = block.group(1).lstrip("\n")
        lines   = payload.splitlines()
        role, body = lines[0].strip(), "\n".join(lines[1:]).strip()

        if role == "system":
            # strip the helper line "Available tools:" if it survived
            body = re.sub(r"^\s*Available tools:\s*", "", body, flags=re.MULTILINE).strip()

        if role == "assistant":
            tool_calls = [ _parse_tool_call(x) for x in TOOLCALL_RE.findall(body) ]
            body_clean = TOOLCALL_RE.sub("", body).strip() or None
            msg = {"role": "assistant", "content": body_clean}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
        else:
            messages.append({"role": role, "content": body})

    return messages, tool_defs



def main():
    # Define tools (same as in your OpenAI example)
    # tools = [
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "get_weather",
    #             "description": "Get the weather in a given location",
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {
    #                     "location": {
    #                         "type": "string",
    #                         "description": "The city and state, e.g. Chicago, IL",
    #                     },
    #                     "unit": {
    #                         "type": "string",
    #                         "enum": ["celsius", "fahrenheit"],
    #                         "description": "The unit of temperature (celsius or fahrenheit)",
    #                     },
    #                 },
    #                 "required": ["location"],
    #             },
    #         },
    #     }
    # ]
    tools = []
    # TODO: add tools, se no sono dichiarati i tools nel sistema, non vengono mostrati? come anthropic

    # Initial conversation
    tools_example = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": 'This function retrieves relevant data from the available knowledge base. Every term encapsulated like this "term" will force the search to match every term in the text, e.g. the query \'Trovare "giacenza media" conto\' will return only the documents that contain "giacenza media".',
                "parameters": {
                    "type": "object",
                    "required": ["queries"],
                    "properties": {
                        "queries": {
                            "type": "array", # Type info can be added to formatter/parser
                            "description": "The list of chosen queries. (3 queries for best results)",
                            # Items type info missing in original, assume string for parsing
                            "items": {"type": "string"}
                        }
                    },
                },
            },
        },
         { # Add another tool for testing
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. Chicago, IL",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature (celsius or fahrenheit)",
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = {"messages": [
             {
                "role": "system",
                "content": "You are a helpful hotel assistant focused on guest experience.",
            },
            {
                "role": "user",
                "content": "Buongiorno, durante il mio ultimo soggiorno ho notato alcune difficoltà legate all'esperienza degli ospiti, in particolare riguardo al check-in e alla pulizia della camera. Potreste fornirmi maggiori informazioni su come migliorate la guest experience per garantire un soggiorno più confortevole?",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_JHq3GlDaGx9vjUElzFhv3Thk",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"queries": ["\\"migliorare guest experience\\" hotel", "\\"difficolt\\u00e0 check-in\\" \\"pulizia camera\\"", "\\"comfort soggiorno\\" soluzioni hotel"]}',
                        },
                    }
                ],
            },
             { # Add a tool result message
                "role": "tool",
                "tool_call_id": "call_JHq3GlDaGx9vjUElzFhv3Thk",
                "content": '{"results": ["Article about optimizing check-in flow.", "Guide to enhanced room cleaning protocols."]}'
             },
             { # Add a final assistant response
                 "role": "assistant",
                 "content": "Grazie per il suo feedback. Abbiamo implementato un processo di check-in digitale e rafforzato i nostri standard di pulizia con controlli aggiuntivi per migliorare il comfort e la sicurezza."
             }
        ],
        "functions": tools_example
    }

    print("\n===== COMPLETE CONVERSATION =====")
    print(format_with_chatml(messages["messages"], tools=messages["functions"]))
    print(parse_chatml(format_with_chatml(messages["messages"], tools=messages["functions"])))


if __name__ == "__main__":
    main()

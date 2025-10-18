import logging
import asyncio
from dataclasses import dataclass
import numpy as np
from hashlib import md5
from functools import wraps
from typing import Any
import tiktoken
import re
import os
import html
import json
import numbers
logger = logging.getLogger("multimodal-graphrag")

ENCODER = None

@dataclass
class EmbeddingFunc:
    """
    Define a function class for embedding.

    Parameters:
    - embedding_dim: Dimension of the embedding vector
    - max_token_size: Maximum token size
    - func: Callable object for executing embedding operations
    """
    embedding_dim: int
    max_token_size: int
    func: callable
    async def __call__(self, *args, **kwargs) -> np.ndarray:
        """
        Call the embedding function and return results.

        Parameters:
        - *args: Positional arguments
        - **kwargs: Keyword arguments

        Returns:
        - np.ndarray: Embedding results
        """
        return await self.func(*args, **kwargs)

# -----------------------------------------------------------------------------------
# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input
    # Remove HTML escape characters
    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    # Remove control characters and other unwanted characters
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)

def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]

def pack_user_ass_to_openai_messages(*args: str):
    """
    Pack user and assistant conversation into OpenAI message format.

    This function accepts a series of string arguments and wraps them into alternating
    user and assistant role messages. This is particularly useful for converting conversation
    history into a format that can be processed by OpenAI's API. Used for history in _op.py.

    Parameters:
    *args (str): One or more string arguments representing alternating statements between
                 user and assistant in the conversation.

    Returns:
    list: A list of dictionaries, each containing two key-value pairs:
        - 'role': Indicates the role of the message sender, alternating between 'user'
                  or 'assistant' based on position in the argument sequence.
        - 'content': The message content sent by the sender, from the corresponding position
                     in the input argument sequence.
    """
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]
# Calculate hash value of arguments
def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()

# Calculate md5 hash value
def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()

def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro

# Use specified model to encode string and return token count
def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens

# Use specified model to decode string and return token count
def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content

# Determine if it's a floating point number
def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))

def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """Add restriction of maximum async calling times for a async func"""
    """
    Add maximum concurrent call limit for async functions.

    Parameters:
    - max_size: int, maximum number of concurrent calls allowed.
    - waitting_time: float, check interval time (seconds) when max concurrent calls reached, default 0.0001 seconds.

    Returns:
    - Returns a decorator function for wrapping async functions that need concurrent call limits.
    """
    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            # If current call count reaches max, wait for a period before checking again
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro

# Write json object to file
def write_json(json_obj, file_name):
    with open(file_name, "w", encoding='utf-8') as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)

# Load json object from file
def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name) as f:
        return json.load(f)

import ast

def parse_value(value):
    """
    Parse string value and convert it to appropriate Python type (dict, number, string, etc.).
    """
    try:
        # Use ast.literal_eval to parse dict or other complex structures
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # If cannot parse, return original string
        return value

def read_config_to_dict(file_path):
    config_dict = {}
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            key, value = line.strip().split(',', 1)  # Split only once
            config_dict[key] = parse_value(value)
    
    return config_dict

def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    """
    Truncate list data by token size.

    The purpose of this function is to ensure that the total token count of data in the list
    does not exceed the specified maximum token size. When the total token count exceeds the
    maximum allowed size, the function will return the truncated list.

    Parameters:
    - list_data: list, list to be truncated, where each element is a data item.
    - key: callable, function to extract strings from list data items for calculating token size.
    - max_token_size: int, maximum allowed token size, used to determine truncation point of list data.

    Returns:
    - Truncated list. If max_token_size is less than or equal to 0, returns empty list.

    Note:
    - This function uses tiktoken to encode strings and calculate token count. Please ensure tiktoken
      library is installed before use.
    - Truncation operation is based on the index position where cumulative token count first exceeds
      max_token_size.
    """
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data

# Enclose a string with double quotes
def enclose_string_with_quotes(content: Any) -> str:
    """Enclose a string with quotes"""
    if isinstance(content, numbers.Number):
        return str(content)
    content = str(content)
    content = content.strip().strip("'").strip('"')
    return f'"{content}"'

# Convert multi-dimensional list to CSV format
def list_of_list_to_csv(data: list[list]):
    return "\n".join(
        [
            ",\t".join([f"{enclose_string_with_quotes(data_dd)}" for data_dd in data_d])
            for data_d in data
        ]
    )

def get_latest_graphml_file(folder_path):
    # Regular expression to match the number part in the filename
    pattern = r'graph_merged_image_(\d+)\.graphml'
    
    max_number = -1
    latest_file = None
    namespace = 'chunk_entity_relation'
    file_path = None
    
    # Traverse all files in the folder
    for filename in os.listdir(folder_path):
        # Check if filename matches the target format
        match = re.match(pattern, filename)
        if match:
            # Extract the number part from the filename
            file_number = int(match.group(1))
            # If this number is greater than the current maximum, update max value and filename
            if file_number > max_number:
                max_number = file_number
                namespace = f"merged_image_{max_number}"
                latest_file = filename
                file_path = os.path.join(folder_path, latest_file)
    # If no matching filename found, return default file_path
    if file_path is None:
        file_path = os.path.join(folder_path, 'graph_chunk_entity_relation.graphml')
    return namespace, file_path
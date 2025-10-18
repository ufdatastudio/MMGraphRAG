from openai import AsyncOpenAI, OpenAI
from parameter import EMBED_MODEL, encode, API_KEY, MODEL, URL, MM_API_KEY, MM_MODEL, MM_URL
import numpy as np
import re
import json

from base import wrap_embedding_func_with_attrs,compute_args_hash
from storage import (
    BaseKVStorage,
)

import time
import functools
from openai import RateLimitError

def retry_on_rate_limit(max_retries=3, delay=60):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RateLimitError as e:
                    if attempt == max_retries:
                        raise
                    print(f"Rate limit hit. Retry {attempt}/{max_retries} in {delay} seconds...")
                    time.sleep(delay)
        return wrapper
    return decorator

@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)

async def local_embedding(texts: list[str]) -> np.ndarray:
    return encode(texts)

async def model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_client = OpenAI(
        api_key=API_KEY, base_url=URL
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    # If cache object exists, calculate hash of current request and try to get result from cache
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = openai_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # If cache object exists, store response result in cache
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content

async def multimodel_if_cache(
    user_prompt, img_base, system_prompt, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=MM_API_KEY, base_url=MM_URL
    )
    messages = []
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "system", "content": [
          {
            "type": "text",
            "text": system_prompt
          }
        ]})
    messages.append({"role": "user", "content": [
          {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_base}"},
          },
          {
            "type": "text",
            "text": user_prompt
          }
        ]})
    # If cache object exists, calculate hash of current request and try to get result from cache
    if hashing_kv is not None:
        args_hash = compute_args_hash(MM_MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=MM_MODEL, messages=messages, **kwargs
    )

    # If cache object exists, store response result in cache
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MM_MODEL}}
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content

# Normalization processing function
def normalize_to_json(output):
    # Use regex to extract JSON part
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            # Validate JSON format
            json_obj = json.loads(json_str)
            return json_obj  # Return normalized JSON object
        except json.JSONDecodeError as e:
            print(f"JSON decoding failed: {e}")
            return None
    else:
        print("No valid JSON part found")
        return None

def normalize_to_json_list(output):
    """
    Extract and validate JSON list format string, returning parsed JSON object list.
    Attempts to extract as much content as possible even if JSON is incomplete.
    """
    # Remove escape characters and extra whitespace
    cleaned_output = output.replace('\\"', '"').strip()
    
    # Use lenient regex to extract possible JSON fragments
    match = re.search(r"\[\s*(\{.*?\})*?\s*]", cleaned_output, re.DOTALL)
    
    if match:
        json_str = match.group(0)
        
        # Remove extra commas (possibly caused by truncation)
        json_str = re.sub(r",\s*]", "]", json_str)
        json_str = re.sub(r",\s*}$", "}", json_str)

        try:
            # Try complete parsing
            json_obj = json.loads(json_str)
            if isinstance(json_obj, list):
                return json_obj
        except json.JSONDecodeError:
            # If complete parsing fails, try item-by-item parsing
            print("Complete parsing failed, attempting item-by-item parsing...")
            items = []
            for partial_match in re.finditer(r"\{.*?\}", json_str, re.DOTALL):
                try:
                    item = json.loads(partial_match.group(0))
                    items.append(item)
                except json.JSONDecodeError:
                    print("Skipping invalid JSON fragment")
            return items if items else []
    else:
        print("No valid JSON fragment found")
        return []

# Use LLM to answer
@retry_on_rate_limit()
def get_llm_response(cur_prompt, system_content):
    client = OpenAI(
        base_url=URL, api_key=API_KEY
    )

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_content,
            },
            {"role": "user", "content": cur_prompt},
        ],
    )

    response = completion.choices[0].message.content
    return response

# Call multimodal LLM
def get_mmllm_response(cur_prompt, system_content, img_base):
    client = OpenAI(
        base_url=MM_URL, api_key=MM_API_KEY
    )

    completion = client.chat.completions.create(
        model=MM_MODEL,
        messages=[
            {"role": "system", "content": [
                    {
                        "type": "text",
                        "text": system_content
                    }
                    ]},
            {"role": "user", "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base}"},
                    },
                    {
                        "type": "text",
                        "text": cur_prompt
                    }
                    ]},
        ],
    )

    response = completion.choices[0].message.content
    return response


import re
import asyncio
import os
import json
import base64
import shutil
import subprocess
from typing import Callable, Dict, List, Optional, Type, Union, cast
from dataclasses import dataclass
from parameter import mineru_dir

from base import (
    logger,
    compute_mdhash_id,
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    read_config_to_dict,
)

from storage import (
    BaseKVStorage,
    JsonKVStorage,
    StorageNameSpace,
)
from llm import multimodel_if_cache
from prompt import PROMPTS

def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    """
    Chunk text based on token size.

    This function is used to chunk given text content according to a specified token size limit, 
    while ensuring overlap between adjacent chunks. It is mainly used for processing large texts 
    to adapt to input limitations of models like OpenAI's GPT series.

    Parameters:
    - content: str, the text content to be chunked.
    - overlap_token_size: int, default 128. The number of overlapping tokens between adjacent text chunks.
    - max_token_size: int, default 1024. The maximum number of tokens per text chunk.
    - tiktoken_model: str, default "gpt-4o". The tiktoken model used for tokenization and detokenization.

    Returns:
    - List[Dict[str, Any]], a list containing the number of tokens, text content, and chunk order index for each text chunk.
    """
    # Use the specified tiktoken model to tokenize the text
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    # Initialize list to store chunking results
    results = []
    # Traverse tokens and chunk based on max_token_size and overlap_token_size
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        # Get chunk tokens based on current chunk start position and maximum token count limit
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        # Add current chunk's token count, text content, and chunk order index to results list
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results

@dataclass
class text_chunking_func:
    # Chunking function
    chunk_func: Callable[[str, Optional[int], Optional[int], Optional[str]], List[Dict[str, Union[str, int]]]] = chunking_by_token_size
    # Chunk size
    chunk_token_size: int = 1200
    # Chunk overlap quantity
    chunk_overlap_token_size: int = 100
    

    # Key-value storage, JSON format, specifically defined in storage.py
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    # Get global settings
    cache_path = os.getenv('CACHE_PATH')
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    
    # Tiktoken model name, default is gpt-4o, moonshot-v1-32k is also compatible
    tiktoken_model_name = global_config["tiktoken_model_name"]

    def __post_init__(self):
        # Get global settings
        cache_path = os.getenv('CACHE_PATH')
        global_config_path = os.path.join(cache_path,"global_config.csv")
        global_config = read_config_to_dict(global_config_path)
        # Initialize storage class instance for storing full documents
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config = global_config
        )
        # Initialize storage class instance for storing text chunks
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config = global_config
        )
    
    async def text_chunking(self,string_or_strings):
        try:
            # If input is a string, convert it to a list
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
            # ---------- new docs
            # Strip whitespace from each element in string_or_strings list, use it as document content.
            # Calculate its MD5 hash value and add prefix 'doc-' as key, content itself as value, generate a new dictionary new_docs.
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            # Filter out new document IDs that need to be added
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            # Update new docs dictionary based on filter results
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            # If no new docs need to be added, log and return
            if not len(new_docs):
                logger.warning(f"All docs are already in the storage")
                return
            # Log insertion of new documents
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            # ---------- chunking
            inserting_chunks = {}
            for doc_key, doc in new_docs.items():
                # Generate chunks for each document
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_key,
                    }
                    for dp in self.chunk_func(
                        doc["content"],
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                        tiktoken_model=self.tiktoken_model_name,
                    )
                }
                inserting_chunks.update(chunks)
            # Filter out new chunk IDs that need to be added
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            # Update new chunks dictionary based on filter results
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            # If no new chunks need to be added, log and return
            if not len(inserting_chunks):
                logger.warning(f"All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            # Submit all updates and indexing operations
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._text_chunking_done()
    async def _text_chunking_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

image_description_prompt_user = PROMPTS["image_description_user"]
image_description_prompt_user_examples = PROMPTS["image_description_user_with_examples"]
image_description_prompt_system = PROMPTS["image_description_system"]

async def get_image_description(image_path, caption, footnote, context):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    caption_string = " ".join(caption) 
    footnote_string = " ".join(footnote) 
    user_prompt = image_description_prompt_user_examples.format(caption=caption_string, footnote=footnote_string, context=context)
    result =  await multimodel_if_cache(user_prompt=user_prompt,img_base=base64_image,system_prompt=image_description_prompt_system)
    
    description_match = re.search(r'"description": "([^"]*)"', result)
    segmentation_match = re.search(r'"segmentation": (\w+)', result)
    # Get matched values
    if description_match:
        image_description = description_match.group(1)
    else:
        image_description = "No description."

    if segmentation_match:
        segmentation_str = segmentation_match.group(1)
    else:
        segmentation_str = "false"
    segmentation = True if segmentation_str.lower() == 'true' else False
    return image_description, segmentation

def find_chunk_for_image(text_chunks, context):
    """
    Find the chunk that an image belongs to based on the text before and after the image.
    Prioritize chunks that contain more consecutive characters, ignoring newlines.
    """
    best_chunk_id = None
    best_match_count = 0

    # If the merged text is empty, return None
    if not context:
        return None

    # Traverse all chunks
    for chunk_id, chunk_data in text_chunks.items():
        # Remove newlines from the chunk
        chunk_content = chunk_data['content'].replace('\n', '')

        # Calculate the match degree between combined text and chunk content (based on word matching)
        match_count = sum(1 for word in context.split() if word in chunk_content)

        # If the current chunk has the highest match degree, select it
        if match_count > best_match_count:
            best_match_count = match_count
            best_chunk_id = chunk_id

    return best_chunk_id
def compress_image_to_size(input_image, output_image_path, target_size_mb=5, step=10, quality=90):
    """
    Compress an image to be within the target size (in MB).

    Parameters:
    input_image (PIL.Image): The input image as a PIL object.
    output_image_path (str): The output image path.
    target_size_mb (int): Target size in MB, default is 5MB.
    step (int): The quality step to decrease each time, default is 10.
    quality (int): Initial image save quality, default is 90.

    Returns:
    bool: Whether the compression to within the target size was successful.
    """
    target_size_bytes = target_size_mb * 1024 * 1024  # Convert target size to bytes

    # First save the image and check its size
    img = input_image
    img.save(output_image_path, quality=quality)
    if os.path.getsize(output_image_path) <= target_size_bytes:
        return True

    # Try gradually reducing quality until image size is less than target size
    while os.path.getsize(output_image_path) > target_size_bytes and quality > 10:
        quality -= step
        img.save(output_image_path, quality=quality)
    
    # Check if final size meets requirements
    if os.path.getsize(output_image_path) <= target_size_bytes:
        return True
    else:
        print("Unable to compress the image to within the target size, please adjust the initial quality or step in preprocessing.py.")
        return False

def clear_images_in_md(content):
    # Use regex to find and delete all image content matching the format
    content = re.sub(r'!\[\]\([^)]*\)', '', content)
    return content

def image_move_remove(json_path, target_folder, folder_path):
    # Load JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Counter starts from 1
    img_counter = 1

    # Traverse each entry in the data
    for entry in data:
        if 'img_path' in entry:
            # Get current image path
            img_path = entry['img_path']
            original_img_path = os.path.join(folder_path, img_path)
            # Set new image filename
            new_img_name = f"image_{img_counter}.jpg"
            new_img_path = os.path.join(target_folder, new_img_name)
            
            # Copy and rename image file
            try:
                # Check if original_img_path is a file not a directory
                if os.path.isfile(original_img_path):
                    shutil.copy(original_img_path, new_img_path)
                    print(f"Copied {original_img_path} to {new_img_path}")
                    img_counter += 1  # Increment counter
                else:
                    print(f"Skipped {original_img_path}, because it is a directory.")
            except FileNotFoundError:
                print(f"Image not found: {original_img_path}")

def get_content_list_json_file(folder_path):
    # Traverse files in the specified folder
    for file_name in os.listdir(folder_path):
        # Only select files ending with "_content_list.json"
        if file_name.endswith('_content_list.json'):
            file_path = os.path.join(folder_path, file_name)
    return file_path

def rename_images_in_json(data):
    # Initialize counter
    image_counter = 1
    
    # Traverse JSON data, find all img_path and rename
    for item in data:
        if "img_path" in item and item["img_path"] != "":
            new_image_path = f"images/image_{image_counter}.jpg"
            # Update img_path in JSON
            item["img_path"] = new_image_path
            image_counter += 1
    return data

# Use MinerU to preprocess PDF, save output results in output folder
def pdf2markdown(pdf_path, output_dir):
    # Get PDF filename (without extension)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    mineru_folder = os.path.join(mineru_dir, pdf_name, "auto")
    if os.path.exists(mineru_folder) and any(file.endswith(".md") for file in os.listdir(mineru_folder)):
        logger.info(f"MinerU already finished!")
        return mineru_folder

    output_folder = os.path.join(output_dir, pdf_name, "auto")

    # Check if output_folder contains .md files
    if os.path.exists(output_folder) and any(file.endswith(".md") for file in os.listdir(output_folder)):
        logger.info(f"MinerU already finished!")
        return output_folder

    # Construct and run command
    try:
        logger.info(f"MinerU processing...")
        # Set environment variable for GPU device
        # gpu_id = 0
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # env_name = "MinerU"
        # command = ["conda", "run", "-n", env_name, 'magic-pdf', '-p', pdf_path, '-o', output_dir]
        command = ['magic-pdf', '-p', pdf_path, '-o', output_dir]
        subprocess.run(
            command,
            capture_output=True, text=True, check=True
        )
        logger.info(f"MinerU finished!")
    except subprocess.CalledProcessError as e:
        # If command execution fails, print error message
        print("Error:", e.stderr)
    return output_folder

async def extract_text_and_images_with_chunks(pdf_path, output_dir, context_length):
    """
    Extract text chunks from PDF and associate them with images. Extract and integrate context text before and after images.
    """
    folder_path = pdf2markdown(pdf_path, output_dir)
    files = os.listdir(folder_path)
    markdown_files = [file for file in files if file.endswith(".md")]

    if len(markdown_files) != 1:
        raise ValueError("No unique .md file was found in the folder. Please ensure there is only one .md file in the folder.")

    markdown_file_path = os.path.join(folder_path, markdown_files[0])
    with open(markdown_file_path, 'r', encoding='utf-8') as file:
        full_text = file.read()
    
    full_text = clear_images_in_md(full_text)

    # First instantiate the class
    text_chunking_instance = text_chunking_func()
    
    # Chunk the document text
    await text_chunking_instance.text_chunking(full_text)
    
    filepath = os.path.join(output_dir, 'kv_store_text_chunks.json')
    with open(filepath, 'r') as file:
        text_chunks = json.load(file)

    # Create directory to save images
    images_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    # Read JSON file, rename in order, and move images
    json_file_path = get_content_list_json_file(folder_path)
    image_move_remove(json_file_path, images_dir, folder_path)
    
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    data = rename_images_in_json(data)

    image_data = {}
    image_counter = 1
     # Traverse JSON data and extract image information
    for i, item in enumerate(data):
        if "img_path" in item and item["img_path"] != "":
            # Build image_id
            image_id = image_counter
            image_counter += 1
            
            # Get and modify image_path
            img_path = os.path.join(output_dir, item["img_path"])
            # Get caption and footnote
            if item["type"]=="image":
                caption = item.get("img_caption", [])
                footnote = item.get("img_footnote", [])
            elif item["type"] == "table":
                caption = item.get("table_caption", [])
                footnote = item.get("table_footnote", [])

            # Extract context
            context = ""

            # Extract forward
            word_count = 0
            prev_index = i - 1
            while word_count < context_length and prev_index >= 0:
                prev_text = data[prev_index].get("text", "")
                prev_words = prev_text.split()  # Split previous text into words or characters
                prev_remaining_words = context_length - word_count
                # Take specified number of words or characters from the end
                selected_words = prev_words[-prev_remaining_words:]
                context = " ".join(selected_words) + " " + context
                word_count += len(selected_words)
                prev_index -= 1

            # Extract backward
            word_count = 0
            next_index = i + 1
            while word_count < context_length and next_index < len(data):
                next_text = data[next_index].get("text", "")
                next_words = next_text.split()  # Split following text into words or characters
                next_remaining_words = context_length - word_count
                # Take specified number of words or characters from the beginning
                selected_words = next_words[:next_remaining_words]
                context = " ".join(selected_words) + " " + context
                word_count += len(selected_words)
                next_index += 1

            # Get chunk_order_index and description
            associated_chunk_id = find_chunk_for_image(text_chunks, context)
            description, segmentation  = await get_image_description(img_path, caption, footnote, context)

            image_key = f"image_{image_id}"
            # Build image information dictionary and add to list
            image_data[image_key] = {
                "image_id": image_id,
                "image_path": img_path,
                "caption": caption,
                "footnote": footnote,
                "context": context,
                "chunk_order_index": text_chunks[associated_chunk_id]['chunk_order_index'],
                "chunk_id": associated_chunk_id,
                "description": description,
                "segmentation": segmentation
            }
    return image_data

@dataclass
class chunking_func_pdf2md:
    # Image extraction context length (100 each, so total length is 200)
    context_length: int = 100

    # Key-value storage, JSON format, specifically defined in storage.py
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage

    def __post_init__(self):
        # Get global settings
        cache_path = os.getenv('CACHE_PATH')
        global_config_path = os.path.join(cache_path,"global_config.csv")
        global_config = read_config_to_dict(global_config_path)
        # Initialize storage class instance for storing image properties
        self.image_data = self.key_string_value_json_storage_cls(
            namespace="image_data", global_config = global_config
        )
    
    async def extract_text_and_images(self, pdf_path):
        try:
            # Get global settings
            cache_path = os.getenv('CACHE_PATH')
            global_config_path = os.path.join(cache_path,"global_config.csv")
            global_config = read_config_to_dict(global_config_path)
            output_dir = global_config["working_dir"]
            context_length = self.context_length
            imagedata = await extract_text_and_images_with_chunks(pdf_path, output_dir, context_length)
            # Submit all updates and indexing operations
            await self.image_data.upsert(imagedata)
        finally:
            await self._chunking_done()

    async def _chunking_done(self):
        tasks = []
        for storage_inst in [self.image_data]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
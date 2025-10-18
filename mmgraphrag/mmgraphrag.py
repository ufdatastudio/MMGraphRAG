import asyncio
import os
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from base import logger
from parameter import cache_path
if not os.path.exists(cache_path):
    os.makedirs(cache_path)
os.environ['CACHE_PATH'] = cache_path
from llm import model_if_cache
from parameter import QueryParam

# Return type is asyncio.AbstractEventLoop, the event loop object. Ensures that a valid event loop is returned regardless of whether one already exists in the current environment.
def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        # If there is already an event loop, use it.
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If in a sub-thread, create a new event loop.
        logger.info("Creating a new event loop in a sub-thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

@dataclass
class MMGraphRAG:
    # Working directory path, default is a directory generated based on current date-time
    working_dir: str = field(
        default_factory=lambda: f"./mm_graphrag_output_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # Entity extraction maximum "gleaning" count, i.e., repeated extraction count, default is no repeated extraction
    entity_extract_max_gleaning: int = 1
    # Entity summary maximum tokens
    entity_summary_to_max_tokens: int = 500

    # Provide optional parameter dictionary for vector database storage class
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)

    # LLM related calls
    model_func: callable = model_if_cache
    model_max_token_size: int = 32768

    # Batch size, default is 32
    embedding_batch_num: int = 32

    # Tiktoken model name, default is gpt-4o, most models are compatible
    tiktoken_model_name: str = "gpt-4o"

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    # If node2vec_params is not explicitly passed in, this lambda function is called to automatically generate and assign this default dictionary
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "num_walks": 10,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # Threshold for comparing query result quality
    query_better_than_threshold: float = 0.2

    query_mode: bool = False
    # File loading mode, 0 for docx files, 1 for direct PDF parsing, 2 for pdf2markdown parsing method
    input_mode: int = 2

    cache_path = cache_path

    # This method is called after object initialization, its main purpose is to print configuration information and make adjustments based on configuration
    def __post_init__(self):
        # Print object attributes as key-value pairs for debugging and logging
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GraphRAG init with param:\n\n  {_print_config}\n")
        global_config = asdict(self)

        global_config_path = os.path.join(cache_path,"global_config.csv")
        # Save global_config dictionary to CSV file
        with open(global_config_path, 'w', newline='') as file:
            for key, value in global_config.items():
                file.write(f"{key},{value}\n")
        
        # Ensure working directory exists, create if it doesn't
        if os.path.exists(self.working_dir):
            logger.info(f"Using existing working directory {self.working_dir}")
        else:
            os.makedirs(self.working_dir)
            logger.info(f"Creating working directory {self.working_dir}")
        
        from preprocessing import chunking_func
        from pdf_preprocessing import chunking_func_pdf
        from pdf2md_preprocessing import chunking_func_pdf2md
        from text2graph import extract_entities_from_text
        from query import local_query
        
        # Instantiate classes
        if self.query_mode:
            self.localquery = local_query()
        else:
            self.ChunkingFunc = chunking_func()
            self.ChunkingFunc_pdf = chunking_func_pdf()
            self.ChunkingFunc_pdf2md = chunking_func_pdf2md()
            self.ExtractEntitiesFromText = extract_entities_from_text()
    
    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        response = await self.localquery.local_query(
            query,
            param,
        )
        return response

    def index(self, path):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aindex(path))
        
    async def aindex(self,path):
        from img2graph import img2graph
        if self.input_mode == 0:
            await self.ChunkingFunc.extract_text_and_images(path)
        elif self.input_mode == 1:
            await self.ChunkingFunc_pdf.extract_text_and_images(path)
        elif self.input_mode == 2:
            # Define path to kv_store_image_data.json file
            kv_store_path = os.path.join(self.working_dir, "kv_store_image_data.json")
            if os.path.exists(kv_store_path):
                with open(kv_store_path, "r", encoding="utf-8") as file:
                    content = json.load(file)
                    # Check if JSON file is empty {}
                    if content == {}:
                        logger.info(f"{kv_store_path} exists but is empty. Proceeding with preprocess.")
                        await self.ChunkingFunc_pdf2md.extract_text_and_images(path)
                    else:
                        logger.info(f"{kv_store_path} exists and is not empty. Skipping preprocess.")
            else:
                await self.ChunkingFunc_pdf2md.extract_text_and_images(path)
        filepath = os.path.join(self.working_dir, 'kv_store_text_chunks.json')
        with open(filepath, 'r') as file:
            chunks = json.load(file)
        await self.ExtractEntitiesFromText.text_entity_extraction(chunks)
        imgfolderpath = os.path.join(self.working_dir, 'images')
        await img2graph(imgfolderpath)
        filepath2 = os.path.join(self.working_dir, 'kv_store_image_data.json')
        with open(filepath2, 'r') as file:
            image_data = json.load(file)
        from fusion import fusion, create_EntityVDB
        # Check if image_data is an empty dictionary
        if image_data:
            img_ids = list(image_data.keys())
            await fusion(img_ids)
        else:
            print("No images extracted, skipping fusion operation")
            createvdb = create_EntityVDB()
            await createvdb.create_vdb()
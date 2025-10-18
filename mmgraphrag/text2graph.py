from collections import defaultdict,Counter
from typing import Type, Union, cast
from dataclasses import dataclass
from functools import partial
import asyncio
import re
import os
import json

from prompt import GRAPH_FIELD_SEP, PROMPTS
from base import (
    logger,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    clean_str,
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    is_float_regex,
    limit_async_func_call,
    read_config_to_dict,
)
from storage import (
    BaseGraphStorage,
    StorageNameSpace,
    BaseKVStorage,
    JsonKVStorage,
    NetworkXStorage,
    TextChunkSchema
)

from llm import model_if_cache

# Async function to handle single entity extraction task result
async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    """
    Process single entity extraction task.

    This function validates and processes given entity record attributes, extracting information
    such as entity name, type, and description, and returns a dictionary containing this information
    along with the entity's source identifier.

    Parameters:
    - record_attributes: A list of strings containing entity attribute information. Expected to have
      at least 4 elements, with the first element being 'entity', identifying this as an entity record.
    - chunk_key: A string representing the unique identifier of the entity information source.

    Returns:
    - If record attributes are valid, returns a dictionary containing entity name, type, description,
      and source identifier.
    - If record attributes are invalid (insufficient elements or first element is not 'entity'),
      returns None.
    """
    # Check if record_attributes list has at least 4 elements and first element is 'entity'
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )

async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    """
    Process entity and relationship descriptions and generate summaries based on global config.

    Parameters:
    - entity_or_relation_name: Name of entity or relationship.
    - description: Description of entity or relationship.
    - global_config: Global configuration containing model, token size, summary max tokens, etc.

    Returns:
    - Generated summary or original description.
    """
    # Get corresponding function and parameters from global config
    use_llm_func = model_if_cache
    llm_max_tokens = global_config["model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    # Encode description information
    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    # If description token count is less than max summary tokens, return description directly
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    # Set prompt
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    # Get description suitable for model max token count
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    # Build context base information
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    # Build final prompt
    user_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    # Use language model to generate summary
    # summary = await use_llm_func(user_prompt, max_tokens=summary_max_tokens)
    summary = await use_llm_func(user_prompt, max_completion_tokens=summary_max_tokens)
    return summary

async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )

async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """
    Merge node data and update or insert nodes in the knowledge graph.

    This function first attempts to retrieve existing node information from the knowledge graph,
    then merges it with newly acquired node data. The merged node data will be updated or inserted
    into the knowledge graph according to given rules.

    Parameters:
    - entity_name (str): Entity name, used to identify nodes in the knowledge graph.
    - nodes_data (list[dict]): A set of node data, where each node data is a dictionary.
    - knwoledge_graph_inst (BaseGraphStorage): Knowledge graph instance for operating the knowledge graph.
    - global_config (dict): Global configuration information, possibly used for configuring merge or
                            processing rules for node data.

    Returns:
    - node_data (dict): Updated or inserted node data.
    """
    # Initialize lists to store existing node information
    already_entitiy_types = []
    already_source_ids = []
    already_description = []
    # Get existing node information from knowledge graph
    already_node = await knwoledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        # If node exists, add existing information to corresponding lists
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])
    # Merge entity types of new and old nodes, select the most frequent one as new entity type
    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    # Merge descriptions of new and old nodes, connect all different descriptions using separator
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    # Merge source IDs of new and old nodes, connect all different source IDs using separator
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    # Process entity description information
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    # Update or insert node into knowledge graph
    await knwoledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    # Add entity name to node data
    node_data["entity_name"] = entity_name
    return node_data

async def _merge_edges_then_upsert(
   
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """
    Merge edge data and insert/update knowledge graph.
    This function checks if an edge exists between src_id and tgt_id. If it exists, retrieves
    current edge data and merges it with new edge data (edges_data); if not, directly inserts
    new edge data. Additionally, if src_id or tgt_id doesn't have corresponding nodes in the graph,
    inserts nodes with default attributes.

    Parameters:
    - src_id (str): Starting node ID of the edge.
    - tgt_id (str): Target node ID of the edge.
    - edges_data (list[dict]): Contains data for one or more edges.
    - knwoledge_graph_inst (BaseGraphStorage): Knowledge graph instance.
    - global_config (dict): Global configuration.
    """
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    # If edge exists between src_id and tgt_id, get existing edge data
    if await knwoledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knwoledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_order.append(already_edge.get("order", 1))

    # [numberchiffre]: `Relationship.order` is only returned from DSPy's predictions
    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
     # Calculate total weight
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
     # Merge and deduplicate descriptions, sort and convert to string
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    # Merge and deduplicate source IDs
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    # Ensure src_id and tgt_id have nodes in the graph, insert if they don't exist
    for need_insert_id in [src_id, tgt_id]:
        if not (await knwoledge_graph_inst.has_node(need_insert_id)):
            await knwoledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    # Insert/update edge between src_id and tgt_id
    await knwoledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight, description=description, source_id=source_id, order=order
        ),
    )

async def extract_entities(
    cache_dir: BaseKVStorage,
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    """
    Async function extract_entities extracts entities from text chunks and updates knowledge graph.

    Parameters:
    chunks (dict[str, TextChunkSchema]): Text chunk dictionary, keys are text chunk identifiers,
                                         values are TextChunkSchema objects containing text chunk content.
    knwoledge_graph_inst (BaseGraphStorage): Knowledge graph instance for storing extracted entities and relationships.
    entity_vdb (BaseVectorStorage): Entity vector database instance for storing vector representations of entities.
    global_config (dict): Global configuration dictionary containing model functions, max iterations, and other parameters.

    Returns:
    Union[BaseGraphStorage, None]: Updated knowledge graph instance, returns None if no entities extracted.
    """
    # Store entities and relationships for each chunk
    output_json_path = f"{global_config['working_dir']}/kv_store_chunk_knowledge_graph.json"

    model_max_async: int = 16
    # Limit async call count for model function and configure hash key-value storage
    use_llm_func = limit_async_func_call(model_max_async)(
        partial(model_if_cache, hashing_kv=cache_dir)
    )
    # Get entity extraction max gleaning iterations from global config
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    # Sort text chunks for sequential processing
    ordered_chunks = list(chunks.items())
    # Prepare entity extraction prompt template and context base information
    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entity_continue_extraction"]
    if_loop_prompt = PROMPTS["entity_if_loop_extraction"]
    # Initialize counters to track processed text chunks, extracted entities, and relationships
    already_processed = 0
    already_entities = 0
    already_relations = 0
    # Implement a global result dictionary to correspond with json format
    chunk_knowledge_graph_info = {}

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """
        Async function _process_single_content processes content of a single text chunk, extracts entities and updates knowledge graph.

        Parameters:
        chunk_key_dp (tuple[str, TextChunkSchema]): Text chunk key-value pair containing text chunk identifier and content.

        Returns:
        dict: Dictionary containing possible nodes and edges extracted from text chunk.
        """
        # Initialize nonlocal variables for tracking processing progress and statistics
        nonlocal already_processed, already_entities, already_relations
        # Parse text chunk key and data
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        chunk_order_index = chunk_dp["chunk_order_index"]
        # Build prompt and call model function to extract entities
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)
        # Build conversation history and perform multiple iterations to extract more entities
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        # Loop up to entity_extract_max_gleaning times for iterative extraction
        for now_glean_index in range(entity_extract_max_gleaning):
            # Call model function to continue extraction
            glean_result = await use_llm_func(continue_prompt, history_messages=history)
            # Add new results to conversation history and final results
            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            # Check if should continue iteration
            if now_glean_index == entity_extract_max_gleaning - 1:
                break
            # Call model function to check if need to continue iterative extraction
            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            # Parse model return value, remove extra quotes and whitespace, convert to lowercase
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            # If return value is not "yes", exit loop
            if if_loop_result != "yes":
                break
        # Split final extracted results into records using multiple delimiters
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )
        # Use defaultdict to store possible nodes and edges
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)

        # Initialize result dictionary for current text chunk
        chunk_results = {
            "chunk_key": chunk_key,
            "entities": [],
            "relationships": [],
        }
        # Traverse each record to extract nodes and edges
        for record in records:
            # Use regex to extract tuple data from record
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            # Split record attributes using tuple delimiter
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            # Process entity extraction
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                chunk_results["entities"].append(if_entities)
                continue
            # Process relationship extraction
            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
                chunk_results["relationships"].append(if_relation)
        # Update processing progress and statistics
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        # Store results for this text chunk
        # Store results for this text chunk
        chunk_knowledge_graph_info[chunk_order_index] = chunk_results
        return dict(maybe_nodes), dict(maybe_edges)
    # Process all text chunks concurrently
    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()  # clear the progress bar
    # Save knowledge graph information as JSON file
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(chunk_knowledge_graph_info, json_file, ensure_ascii=False, indent=4)
    # Aggregate all possible nodes and edges
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            # When building undirected graph, sort edge nodes in dictionary order
            maybe_edges[tuple(sorted(k))].extend(v)
    # Merge node data and update knowledge graph
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )

    # Merge edge data and update knowledge graph
    await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knwoledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    # If no entities extracted, issue warning and return None
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    return knwoledge_graph_inst

@dataclass
class extract_entities_from_text:
    # Entity extraction function
    text_entity_extraction_func: callable = extract_entities

    # Storage type settings
    # Key-value storage, json, specifically defined in storage.py
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    # Graph database storage, defaults to NetworkXStorage
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage

    def __post_init__(self):
        # Get global settings
        cache_path = os.getenv('CACHE_PATH')
        global_config_path = os.path.join(cache_path,"global_config.csv")
        global_config = read_config_to_dict(global_config_path)
        # Initialize LLM response cache according to config
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=global_config
            )
        )
        # Initialize graph storage class instance for storing chunk entity relationship graph
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=global_config
        )
    async def text_entity_extraction(self, inserting_chunks):
        try:
            # ---------- extract/summary entity and upsert to graph
            # Extract new entities and relationships, and update to knowledge graph
            # Get global settings
            cache_path = os.getenv('CACHE_PATH')
            global_config_path = os.path.join(cache_path,"global_config.csv")
            global_config = read_config_to_dict(global_config_path)
            logger.info("[Entity Extraction]...")
            maybe_new_kg = await self.text_entity_extraction_func(
                self.llm_response_cache,
                inserting_chunks,
                knwoledge_graph_inst=self.chunk_entity_relation_graph,
                global_config=global_config,
            )
            if maybe_new_kg is None:
                logger.warning("No new entities found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg
        finally:
            await self._text_entity_extraction_done()
    async def _text_entity_extraction_done(self):
        tasks = []
        for storage_inst in [
            self.llm_response_cache,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks) 
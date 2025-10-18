from dataclasses import dataclass, field
from typing import Type
import asyncio
import json
import os
import base64
import re

from llm import model_if_cache, local_embedding, multimodel_if_cache
from prompt import PROMPTS, GRAPH_FIELD_SEP
from storage import (
    BaseGraphStorage,
    BaseVectorStorage,
    BaseKVStorage,
    TextChunkSchema,
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from base import (
    logger, 
    truncate_list_by_token_size, 
    split_string_by_multi_markers,
    list_of_list_to_csv,
    read_config_to_dict,
    limit_async_func_call,
    get_latest_graphml_file,
    EmbeddingFunc
)

# Query parameter class, used to define various configuration options during querying.
from parameter import QueryParam

def path_check(path):
    cache_path = os.getenv('CACHE_PATH')
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    working_dir = global_config['working_dir']
    pattern = r'.*(/images/image_\d+.jpg)'
    if not os.path.exists(path):
        path = re.sub(pattern, rf'{working_dir}\1', path)
    return path

def img_path2chunk_id(data,img_data):
    # Build mapping from image_path to chunk_id
    path_to_chunk = {v["image_path"]: v["chunk_id"] for v in img_data.values()}

    # Traverse original data dictionary and perform replacement and deduplication
    for key, value_set in data.items():
        updated_values = set()
        for value in value_set:
            if value.endswith('.jpg'):  # Check if it's a jpg file path
                # Replace with corresponding chunk_id
                chunk_id = path_to_chunk.get(value)
                if chunk_id:
                    updated_values.add(chunk_id)
            else:
                updated_values.add(value)  # Keep original value
        # Update dictionary values and deduplicate (automatically deduplicated by using set)
        data[key] = updated_values
    return data

async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    """
    Asynchronous function: Find the most related text units from entities
    Based on node data and query parameters, find and return the most related text units from the text chunk database and knowledge graph instance

    Parameters:
    - node_datas: List of node data, each node data is a dictionary
    - query_param: Query parameter object, containing information such as local maximum token size
    - text_chunks_db: Text chunk storage object, used to get text chunk data from database
    - knowledge_graph_inst: Knowledge graph storage instance, used to get node and edge information

    Returns:
    - all_text_units: List of all related text units, sorted by relevance and order
    """
    # Split text units based on source IDs from node data
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
        if "source_id" in dp
    ]
    # Concurrently get edge information for all nodes
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    # Collect all one-hop nodes
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    # Concurrently get data for all one-hop nodes
    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )
    # Build text unit mapping for one-hop nodes
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v
    }
    # Normalize text unit mapping for one-hop nodes based on image information
    cache_path = os.getenv('CACHE_PATH')
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    path = os.path.join(global_config['working_dir'], 'kv_store_image_data.json')
    with open(path, 'r') as file:
        image_data = json.load(file)
    all_one_hop_text_units_lookup = img_path2chunk_id(all_one_hop_text_units_lookup,image_data)
    # Initialize full mapping of text units
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if not c_id.startswith('chunk-'):
                continue
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,
            }
    # Check for missing text chunks and log warning
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    # Build list of all text units and sort by order and relation counts
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    # Truncate text unit list based on token size
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.local_max_token_for_text_unit,
    )
    # Return list of text unit data
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]
    return all_text_units

async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    """
    Asynchronous function: Find the most related edges from entities

    Based on the provided node data, find the most related edges from the knowledge graph and sort by relevance

    Parameters:
    node_datas: list[dict] - List of node data, each node data is a dictionary
    query_param: QueryParam - Query parameters, used to limit the token size of returned data
    knowledge_graph_inst: BaseGraphStorage - Knowledge graph storage instance

    Returns:
    list - List of related edge data, sorted by relevance and weight in descending order
    """
    # Concurrently get all edges related to nodes
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    # Create a set to store unordered pairs of all related edges
    all_edges = set()
    for this_edges in all_related_edges:
        # Convert edges to sorted tuples to ensure edge order doesn't affect subsequent processing
        all_edges.update([tuple(sorted(e)) for e in this_edges])
    # Convert set to list for subsequent processing
    all_edges = list(all_edges)
    # Concurrently get detailed information for all edges
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    # Concurrently get relevance for all edges
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    # Build edge data list based on obtained edge information, relevance, and edge metadata
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    # Truncate edge data list based on token size limit in query parameters
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.local_max_token_for_local_context,
    )
    return all_edges_data

async def _build_local_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    """
    Asynchronous function to build local query context.

    This function accepts a query statement and multiple data storage objects, and builds a local query context based on query parameters.
    It queries the top_k most relevant entities from the entity vector database and uses the knowledge graph storage to obtain detailed information, text fragments, and relationships for these entities.

    Parameters:
    - query: Query statement.
    - knowledge_graph_inst: Knowledge graph instance, used to get node and edge information.
    - entities_vdb: Entity vector storage, used to query related entities.
    - text_chunks_db: Text chunk storage, used to find associated text fragments.
    - query_param: Query parameters, containing query options such as top_k.

    Returns:
    - A string containing the query result context in CSV format, or None if no results are found.
    """
    # Query top_k most relevant entities from entity vector database based on query statement
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return "", ""
    # Get node data for queried entities
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    # Get node degrees for queried entities
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    # Integrate node data, including entity name, rank, and node degree
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(
        f"Using {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    # Build CSV data for entities section
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)
    # Build CSV data for relationships section
    relations_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)
    # Build CSV data for text units section
    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    context = f"""
    -----Entities-----
    ```csv
    {entities_context}
    ```
    -----Relationships-----
    ```csv
    {relations_context}
    ```
    -----Sources-----
    ```csv
    {text_units_context}
    ```
    """
    return entities_context, context

@dataclass
class local_query:
    # Use local embedding by default
    embedding_func: EmbeddingFunc = field(default_factory=lambda: local_embedding)
    # Maximum number of concurrent requests
    embedding_func_max_async: int = 16
    # Key-value storage, json format, defined in storage.py
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    # Vector database storage, defined in storage.py
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    # Graph database storage, default is NetworkXStorage
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage

    def __post_init__(self):
        # Get global configuration
        cache_path = os.getenv('CACHE_PATH')
        global_config_path = os.path.join(cache_path,"global_config.csv")
        global_config = read_config_to_dict(global_config_path)
        # Load stored text chunks
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=global_config
        )
        # Load knowledge graph
        kg_namespace, _ = get_latest_graphml_file(global_config['working_dir'])
        self.knowledge_graph_merged = self.graph_storage_cls(
            namespace=kg_namespace, global_config=global_config
        )
        # Limit async calls to embedding function
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        # Load entity vector database
        self.entities_database = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=global_config,
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
        )

    async def local_query(
        self,
        query,
        query_param: QueryParam,
    ) -> str:
        """
        Execute local query based on query and configuration, and return results.

        This function first builds a local query context based on the query and a series of storage instances.
        If the context is empty, it returns a predefined query failure response.
        Otherwise, it constructs a system prompt based on the context and query parameters to generate a response.

        Parameters:
            query: Query string.
            query_param: Query parameters, containing information such as query type and response type.

        Returns:
            Response string generated based on query and context.
        """
        # Get index results
        
        knowledge_graph_inst = self.knowledge_graph_merged

        entities_vdb = self.entities_database

        text_chunks_db = self.text_chunks
        # Get best model function from global configuration
        use_model_func = model_if_cache
        use_multimodel_func = multimodel_if_cache
        # Build local query context
        entities_context, context = await _build_local_query_context(
            query,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
        # Save context string to CSV file
        cache_path = os.getenv('CACHE_PATH')
        global_config_path = os.path.join(cache_path,"global_config.csv")
        global_config = read_config_to_dict(global_config_path)
        context_path = os.path.join(global_config["working_dir"], "context.csv")
        with open(context_path, 'a', newline='') as file:
            file.write(f"{context}\n")
        # If context is empty, return query failure response
        if context is None:
            return PROMPTS["fail_response"]
        # Construct system prompt based on predefined template, including context data and response type from query parameters
        sys_prompt_temp = PROMPTS["local_rag_response_augmented"]
        sys_prompt = sys_prompt_temp.format(
            context_data=context, response_type=query_param.response_type
        )
        # Generate query response
        response = await use_model_func(
            query,
            system_prompt=sys_prompt,
        )
        with open(context_path, 'a', newline='') as file:
            file.write(f"{response}\n")
        # Get multimodal entities
        img_entities = []
        for line in entities_context.split("\n")[1:]:  # Skip the first line header
            parts = line.split(",")
            if len(parts) >= 3 and parts[2].strip().strip('"') == "ORI_IMG":
                entity = parts[1].strip().strip('"')
                img_entities.append(entity)
        img_entities = [entity.lower() for entity in img_entities][:QueryParam.number_of_mmentities]
        logger.info(f'Using multimodal entities{img_entities}')
        if not img_entities:
            return response

        image_data_path = os.path.join(global_config["working_dir"], 'kv_store_image_data.json')
        # Load image_data
        with open(image_data_path, 'r', encoding='utf-8') as file:
            image_data = json.load(file)
        # Get image paths
        image_paths = [image_data[entity]['image_path'] for entity in img_entities if entity in image_data]
        captions = [image_data[entity]['caption'] for entity in img_entities if entity in image_data]
        footnotes = [image_data[entity]['footnote'] for entity in img_entities if entity in image_data]
        # Base64 encoding format
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        
        mm_response = []
        format_query = f'Query:{query}'
        for path, caption, footnote in zip(image_paths, captions, footnotes):
            path = path_check(path)
            image_base = encode_image(path)
            information = f"{caption}, {footnote}"
            mm_prompt = PROMPTS["local_rag_response_multimodal"].format(
                context_data=context, 
                response_type=query_param.response_type, 
                image_information=information
            )
            response = await use_multimodel_func(
                format_query,
                img_base=image_base,
                system_prompt=mm_prompt,
            )
            mm_response.append(response)
        with open(context_path, 'a', newline='') as file:
            file.write(f"mm_response:\n{mm_response}\n")
        merge_prompt = PROMPTS["local_rag_response_multimodal_merge"].format(mm_responses=mm_response)
        mm_merged_response = await use_model_func(
            query,
            system_prompt=merge_prompt,
        )
        with open(context_path, 'a', newline='') as file:
            file.write(f"merged_mm_response:\n{mm_merged_response}\n")
        final_prompt = PROMPTS["local_rag_response_merge"].format(response_type=query_param.response_type, mm_response=mm_merged_response, response=response)
        final_response = await use_model_func(
            format_query,
            system_prompt=final_prompt,
        )
        return final_response
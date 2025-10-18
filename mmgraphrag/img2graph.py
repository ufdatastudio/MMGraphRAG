from dataclasses import dataclass
from collections import defaultdict,Counter 
from typing import Type,cast,Union
from functools import partial
from ultralytics import YOLO
from pathlib import Path
from PIL import Image

import numpy as np
import cv2
import asyncio
import os
import base64
import shutil
import re
import json

from prompt import GRAPH_FIELD_SEP,PROMPTS
from llm import multimodel_if_cache,model_if_cache
from base import (
    logger,
    clean_str,
    limit_async_func_call,
    read_config_to_dict,
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    is_float_regex,
    split_string_by_multi_markers,
)
from storage import (
    BaseGraphStorage,
    StorageNameSpace,
    BaseKVStorage,
    JsonKVStorage,
    NetworkXStorage,
)

async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    """
    Handle single entity extraction task.

    This function is responsible for validating and processing the given entity record attributes, extracting information such as entity name, type, and description,
    and returning a dictionary containing this information along with the entity's source identifier.

    Parameters:
    - record_attributes: A list of strings containing entity attribute information. The list is expected to have at least 4 elements,
      with the first element being 'entity', identifying this as an entity record.
    - chunk_key: A string representing the unique identifier of the source of the entity information.

    Returns:
    - If the record attributes are valid, returns a dictionary containing the entity name, type, description, and source identifier.
    - If the record attributes are invalid (such as insufficient number of elements or the first element is not 'entity'), returns None.
    """
    # Check if the record_attributes list has at least 4 elements and if the first element is 'entity'
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
    Handle entity and relationship descriptions based on global configuration and generate summaries.

    Parameters:
    - entity_or_relation_name: Name of the entity or relationship.
    - description: Description of the entity or relationship.
    - global_config: Global configuration containing model, token size, maximum summary tokens, etc.

    Returns:
    - Generated summary or original description.
    """
    # Get corresponding functions and parameters from global configuration
    use_llm_func = model_if_cache
    llm_max_tokens = global_config["model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    # Encode description information
    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    # If the token count of the description is less than the maximum summary tokens, return the description directly
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    # Set prompt
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    # Get description suitable for model's maximum token count
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
    Merge node data and update or insert nodes into the knowledge graph.

    This function first attempts to retrieve existing node information from the knowledge graph, then merges it with newly obtained node data.
    The merged node data will be updated or inserted into the knowledge graph according to the given rules.

    Parameters:
    - entity_name (str): Entity name, used to identify nodes in the knowledge graph.
    - nodes_data (list[dict]): A set of node data, each node data is a dictionary.
    - knwoledge_graph_inst (BaseGraphStorage): Knowledge graph instance, used to operate the knowledge graph.
    - global_config (dict): Global configuration information, may be used to configure rules for merging or processing node data.

    Returns:
    - node_data (dict): Node data after update or insertion.
    """
    # Initialize lists to store existing node information
    already_entitiy_types = []
    already_source_ids = []
    already_description = []
    # Get existing node information from the knowledge graph
    already_node = await knwoledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        # If the node exists, add the existing information to corresponding lists
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])
    # Merge entity types from old and new nodes, select the most frequent as the new entity type
    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    # Merge descriptions from old and new nodes, connect all different descriptions with separator
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    # Merge source IDs from old and new nodes, connect all different source IDs with separator
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
    # Update or insert node to knowledge graph
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
    This function checks if an edge exists between src_id and tgt_id. If it exists, get the current edge data and
    merge it with new edge data (edges_data); if not, directly insert new edge data. Meanwhile,
    if src_id or tgt_id does not have a corresponding node in the graph, insert a node with default attributes.

    Parameters:
    - src_id (str): Starting node ID of the edge.
    - tgt_id (str): Target node ID of the edge.
    - edges_data (list[dict]): Contains one or more edge data.
    - knwoledge_graph_inst (BaseGraphStorage): Knowledge graph instance.
    - global_config (dict): Global configuration.
    """
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    # If an edge exists between src_id and tgt_id, get existing edge data
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

# Extract feature chunks from original image (single image) and save separately
async def extract_feature_chunks(single_image_path):
    cache_path = os.getenv('CACHE_PATH')
    # Get global configuration
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    
    # Save path is under feature_images folder with original image name
    save_dir = f"{global_config['working_dir']}/images/{Path(single_image_path).stem}"
    # Create save path if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path = os.path.join(global_config["working_dir"], 'kv_store_image_data.json')
    # Load image_data, get current_image_data based on image_path
    with open(path, 'r', encoding='utf-8') as file:
        image_data = json.load(file)
    segmentation = False
    for _, value in image_data.items():
        if value.get("image_path") == single_image_path:
            segmentation = value.get("segmentation", False)

    if not segmentation:
        return save_dir
    
    # Load model, this is the default official model with average performance
    yolo_path = os.path.join(cache_path,"yolov8n-seg.pt")
    model = YOLO(yolo_path)
    # Execute prediction
    results = model(single_image_path, device='cpu')

    # Iterate detection results, as the default results output is a list
    for result in results:
        # Copy original image
        img = np.copy(result.orig_img)
        # Get image filename
        img_name = Path(result.path).stem

        # Traverse detected object contours
        for ci, c in enumerate(result):
            # Get object category
            label = c.names[c.boxes.cls.tolist().pop()]

            # Create a blank mask same size as original image to store object's mask contour
            b_mask = np.zeros(img.shape[:2], np.uint8)

            # Get object's contour information and convert to integer type suitable for OpenCV processing
            contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)

            # Use OpenCV to draw contour on mask, filling the contour region
            _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

            # Combine original image with mask to form segmentation result with black background
            mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
            isolated = cv2.bitwise_and(mask3ch, img)
            # Transparent background
            # isolated = np.dstack([img, b_mask]) 

            # Get bounding box information, crop isolated object to its bounding box
            x1, y1, x2, y2 = c.boxes.xyxy[0].cpu().numpy().astype(np.int32)
            iso_crop = isolated[y1:y2, x1:x2]

            # Create complete save path, including save directory, image filename, object label and sequence number
            save_path = Path(save_dir) / f"{img_name}_{label}-{ci}.jpg"

            # Save segmented object as JPG file
            _ = cv2.imwrite(str(save_path),iso_crop)
    return save_dir

async def feature_image_entity_construction(feature_image_path,use_llm_func):
    entities = []

    # Check if folder is empty (no .jpg images)
    if not any(filename.lower().endswith('.jpg') for filename in os.listdir(feature_image_path)):
        return entities  # Return empty list

    feature_prompt_user = PROMPTS["feature_image_description_user"]
    feature_prompt_system = PROMPTS["feature_image_description_system"]
    tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"]
    record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"]
    for filename in os.listdir(feature_image_path):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(feature_image_path, filename)
            # Open image and check dimensions
            with Image.open(image_path) as image:
                width, height = image.size
            if width > 28 and height > 28:
                # Read and encode as Base64
                with open(image_path, "rb") as image_file:
                    img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                
                # Call LLM function to generate description
                description = await use_llm_func(
                    user_prompt=feature_prompt_user,
                    img_base=img_base64,
                    system_prompt=feature_prompt_system
                )
                
                # Build entity string
                entity = f'("entity"{tuple_delimiter}"{filename}"{tuple_delimiter}"img"{tuple_delimiter}"{description}"){record_delimiter}'
                # Replace extra symbols to ensure format matches expected <|> separator format
                entity = entity.replace("('", "(").replace("')", ")")
                entities.append(entity.replace("\n", ""))
            else:
                logger.info(f"Image {image_path} dimensions too small: width={width}, height={height}")
                os.remove(image_path)
    return entities

async def feature_image_relationship_construction(feature_image_path,image_entities,use_llm_func):
    relationships = []

    # Check if folder is empty (no .jpg images)
    if not any(filename.lower().endswith('.jpg') for filename in os.listdir(feature_image_path)):
        return relationships  # Return empty list

    feature_prompt_user = PROMPTS["entity_alignment_user"]
    feature_prompt_system = PROMPTS["entity_alignment_system"]
    context_base_1 = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"]
        )
    prompt_system = feature_prompt_system.format(**context_base_1)

    for filename in os.listdir(feature_image_path):
        if filename.lower().endswith('.jpg'):
            context_base_2 = dict(
                entity_description = image_entities,
                feature_image_name = filename
                )
            prompt_user = feature_prompt_user.format(**context_base_2)
            image_path = os.path.join(feature_image_path, filename)
            with open(image_path, "rb") as image_file:
                img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                relationship = await use_llm_func(
                                    user_prompt = prompt_user,
                                    img_base = img_base64,
                                    system_prompt = prompt_system
                                )
                relationships.append(relationship)
    return relationships

async def extract_entities_from_image(single_image_path,use_llm_func):
    image_entity_extract_prompt = PROMPTS["image_entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )

    image_entity_system = image_entity_extract_prompt.format(**context_base)

    image_entity_user = """
    Please output the results in the format provided in the example.
    Output:
    """
    with open(single_image_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    result = await use_llm_func(
                    user_prompt = image_entity_user,
                    img_base = img_base64,
                    system_prompt = image_entity_system   
            )
    return result

async def entity_of_original_image(image_path,result1,result2):
    # Initialize as list
    result4 = []
    # Get global configuration
    cache_path = os.getenv('CACHE_PATH')
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    path = os.path.join(global_config["working_dir"], 'kv_store_image_data.json')
    tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"]
    record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"]
    # Load image_data, get current_image_data based on image_path
    with open(path, 'r', encoding='utf-8') as file:
        image_data = json.load(file)
    for image_key, image_info in image_data.items():
        if image_info['image_path'] == image_path:
            current_image_data = image_info
            filename = image_key
            break
    # Normalize to entity variable based on current_image_data
    description = current_image_data["description"]
    entity = f'("entity"{tuple_delimiter}"{filename}"{tuple_delimiter}"ori_img"{tuple_delimiter}"{description}"){record_delimiter}'
    # Replace extra symbols to ensure format matches expected <|> separator format
    entity = entity.replace("('", "(").replace("')", ")")
    result4.append(entity.replace("\n",""))
    # Normalize relationship variable based on result1
    entity_name_pattern = r'\"([^\"]+?\.jpg)\"'
    for entity_feature in result1:
        entity_name = re.findall(entity_name_pattern, entity_feature)[0]
        if entity_name:
            relationship1 = f'("relationship"{tuple_delimiter}"{entity_name}"{tuple_delimiter}"{filename}"{tuple_delimiter}"{entity_name} is an image feature block of {filename}."{tuple_delimiter}10){record_delimiter}'
            result4.append(relationship1)
    # Normalize relationship variable based on result2
    entity_name_pattern2 = r'\"entity\"<\|>\"([^\"]+?)\"'
    entity_names = re.findall(entity_name_pattern2, result2)
    for entity_name2 in entity_names:
        relationship2 = f'("relationship"{tuple_delimiter}"{entity_name2}"{tuple_delimiter}"{filename}"{tuple_delimiter}"{entity_name2} is an entity extracted from {filename}."{tuple_delimiter}10){record_delimiter}'
        result4.append(relationship2)
    return result4

def format_result(result):
    pattern = r'\("entity"<\|>"([^"]+)"<\|>"[^"]*"<\|>"([^"]+)"\)'
    entities = re.findall(pattern, result)
    formatted_result = "\n".join([f'"{entity}"-"{description}"' for entity, description in entities])
    return formatted_result

async def extract_entities(
    cache_dir: BaseKVStorage,
    image_path: str,
    feature_image_path:str,
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    model_max_async :int = 16
    # Limit async calls to model function and configure hashing key-value storage
    use_llm_func = limit_async_func_call(model_max_async)(
        partial(multimodel_if_cache, hashing_kv=cache_dir)
    )
    
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )

    result1 = await feature_image_entity_construction(feature_image_path,use_llm_func)
    result2 = await extract_entities_from_image(image_path,use_llm_func)
    formatted_result2 = format_result(result2)
    result3 = await feature_image_relationship_construction(feature_image_path,formatted_result2,use_llm_func)
    result4 = await entity_of_original_image(image_path,result1,result2)
    final_result = "\n" + "\n".join(result1 + result3 + result4) + result2.strip()

    # Split final extracted results into records using multiple separators
    records = split_string_by_multi_markers(
        final_result,
        [context_base["record_delimiter"], context_base["completion_delimiter"]],
    )
    # Use defaultdict to store possible nodes and edges
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)

    # Initialize result dictionary for current image
    image_results = {
        "image_path": image_path,
        "entities": [],
        "relationships": [],
    }
    # Traverse each record to process node and edge extraction
    for record in records:
        # Use regex to extract tuple data from record
        record = re.search(r"\((.*)\)", record)
        if record is None:
            continue
        record = record.group(1)
        # Split record attributes by tuple separator
        record_attributes = split_string_by_multi_markers(
            record, [context_base["tuple_delimiter"]]
        )
        # Handle entity extraction
        if_entities = await _handle_single_entity_extraction(
            record_attributes, image_path
        )
        if if_entities is not None:
            maybe_nodes[if_entities["entity_name"]].append(if_entities)
            image_results["entities"].append(if_entities)
            continue
        # Handle relationship extraction
        if_relation = await _handle_single_relationship_extraction(
            record_attributes, image_path
        )
        if if_relation is not None:
            maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                if_relation
            )
            image_results["relationships"].append(if_relation)
    m_nodes = defaultdict(list)
    m_edges = defaultdict(list)
    for k, v in maybe_nodes.items():
        m_nodes[k].extend(v)
    for k, v in maybe_edges.items():
        # When building undirected graph, sort edge nodes by lexicographic order
        m_edges[tuple(sorted(k))].extend(v)
    # Merge node data and update knowledge graph
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config)
            for k, v in m_nodes.items()
        ]
    )
    # Merge edge data and update knowledge graph
    await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knwoledge_graph_inst, global_config)
            for k, v in m_edges.items()
        ]
    )
    # If no entities were extracted, issue warning and return None
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    return knwoledge_graph_inst

@dataclass
class extract_entities_from_single_image:
    # Entity extraction function
    image_entity_extraction_func: callable = extract_entities

    # Storage type settings
    # Key-value storage, json format, defined in storage.py
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    # Graph database storage, default is NetworkXStorage
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage

    def __post_init__(self):
        # Get global configuration
        cache_path = os.getenv('CACHE_PATH')
        global_config_path = os.path.join(cache_path,"global_config.csv")
        global_config = read_config_to_dict(global_config_path)

        # Initialize LLM response cache based on configuration
        self.multimodel_llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="multimodel_llm_response_cache", global_config=global_config
            )
        )
        # Initialize graph storage class instance for storing chunk entity relation graph
        self.image_entity_relation_graph = self.graph_storage_cls(
            namespace="image_entity_relation", global_config=global_config
        )
    async def single_image_entity_extraction(self, image_path):
        try:
            # Get global configuration
            cache_path = os.getenv('CACHE_PATH')
            global_config_path = os.path.join(cache_path,"global_config.csv")
            global_config = read_config_to_dict(global_config_path)
            # YOLO image segmentation
            feature_image_path = await extract_feature_chunks(image_path)
            # ---------- extract/summary entity and upsert to graph
            # Extract new entities and relationships, and update to knowledge graph
            logger.info("[Entity Extraction]...")
            maybe_new_kg = await self.image_entity_extraction_func(
                self.multimodel_llm_response_cache,
                image_path,
                feature_image_path,
                knwoledge_graph_inst=self.image_entity_relation_graph,
                global_config=global_config,
            )
            if maybe_new_kg is None:
                logger.warning("No entities found")
                return
            self.image_entity_relation_graph = maybe_new_kg
        finally:
            await self._single_image_entity_extraction_done()
    async def _single_image_entity_extraction_done(self):
        tasks = []
        for storage_inst in [
            self.multimodel_llm_response_cache,
            self.image_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

async def img2graph(image_path):
    # Get global configuration
    cache_path = os.getenv('CACHE_PATH')
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    # Get all .jpg files in directory
    jpg_files = [f for f in os.listdir(image_path) if f.endswith('.jpg')]
    # Complete paths
    jpg_file_paths = [os.path.join(image_path, f) for f in jpg_files]
    for single_image_path in jpg_file_paths:
        extraction = extract_entities_from_single_image()
        if single_image_path.lower().endswith(('.jpg')):
            await extraction.single_image_entity_extraction(single_image_path)
            # Save path is under feature_images folder with original image name
            destination_dir = f"{global_config['working_dir']}/images/{Path(single_image_path).stem}/graph_{Path(single_image_path).stem}_entity_relation.graphml"
            source = f"{global_config['working_dir']}/graph_image_entity_relation.graphml"
            shutil.move(source,destination_dir)
    return
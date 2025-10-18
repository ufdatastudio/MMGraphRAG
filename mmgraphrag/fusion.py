from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass, field
from typing import Type, cast
import math
import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from prompt import PROMPTS, GRAPH_FIELD_SEP
from base import logger

import networkx as nx
import xml.etree.ElementTree as ET
import json
import asyncio
import os
import base64

from llm import local_embedding, get_llm_response, get_mmllm_response, normalize_to_json, normalize_to_json_list
from parameter import encode
from storage import (
    StorageNameSpace,
    BaseVectorStorage,
    NanoVectorDBStorage,
)
from base import (
    limit_async_func_call,
    read_config_to_dict,
    compute_mdhash_id,
    get_latest_graphml_file,
    EmbeddingFunc,
)
cache_path = os.getenv('CACHE_PATH')

# Get global settings
def get_image_data():
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    working_dir = global_config['working_dir']
    json_file_path1 = os.path.join(working_dir, 'kv_store_image_data.json')
    with open(json_file_path1, 'r', encoding='utf-8') as f:
        image_data = json.load(f)
    return image_data

def get_chunk_knowledge_graph():
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    working_dir = global_config['working_dir']
    json_file_path2 = os.path.join(working_dir, 'kv_store_chunk_knowledge_graph.json')
    with open(json_file_path2, 'r', encoding='utf-8') as f:
        chunk_knowledge_graph = json.load(f)
    return chunk_knowledge_graph

def get_text_chunks():
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    working_dir = global_config['working_dir']
    json_file_path3 = os.path.join(working_dir, 'kv_store_text_chunks.json')
    with open(json_file_path3, 'r', encoding='utf-8') as f:
        text_chunks = json.load(f)
    return text_chunks

def extract_entities_from_graph(graphml_file: str) -> list[dict]:
    # Load GraphML file
    graph = nx.read_graphml(graphml_file)
    
    # Store list of all entity nodes
    entity_list = []
    
    # Traverse each node in the graph to extract required data
    for node_id, node_data in graph.nodes(data=True):
        # Extract entity information
        entity_info = {
            "entity_type": node_data.get('entity_type', 'UNKNOWN'),
            "description": node_data.get('description', ''),
            "source_id": node_data.get('source_id', ''),
            "entity_name": node_id
        }
        
        # Add entity information to list
        entity_list.append(entity_info)
    
    return entity_list

@dataclass
class create_EntityVDB:
    # Default to using local embedding
    embedding_func: EmbeddingFunc = field(default_factory=lambda: local_embedding)
    # Maximum number of concurrent requests
    embedding_func_max_async: int = 16
    
    # Vector database storage, specifically defined in storage.py
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage


    def __post_init__(self):
        global_config_path = os.path.join(cache_path,"global_config.csv")
        global_config = read_config_to_dict(global_config_path)
        # Limit async call count for embedding function
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        # Initialize vector database storage class instance according to configuration for storing entities
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=global_config,
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
        )
    async def create_vdb(self):
        global_config_path = os.path.join(cache_path,"global_config.csv")
        global_config = read_config_to_dict(global_config_path)
        working_dir = global_config['working_dir']
        _, graph_file = get_latest_graphml_file(working_dir)
        all_entities_data = extract_entities_from_graph(graph_file)
        # If entity vector database is not empty, store extracted entities into the vector database
        if self.entities_vdb is not None:
            # Construct entity data into vector database format
            data_for_vdb = {
                compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                    "content": dp["entity_name"] + dp["description"],
                    "entity_name": dp["entity_name"],
                }
                for dp in all_entities_data
            }
            # Insert data into vector database
            await self.entities_vdb.upsert(data_for_vdb)
        return await self._create_vdb_done()
    async def _create_vdb_done(self):
        tasks = []
        for storage_inst in [
            self.entities_vdb,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

def get_nearby_chunks(data, index):
    # Get the range of two numbers before and after
    start_index = max(0, index - 1)  # If 0, only take 0 and 1
    end_index = min(len(data) - 1, index + 1)  # If it's the last number, only take itself and the previous one
    
    all_index = list(range(start_index, end_index + 1))
    nearby_chunks = []
    for key, value in data.items():
            if value.get("chunk_order_index") in all_index:
                nearby_chunks.append(value.get("content"))
    return nearby_chunks

def get_nearby_entities(data, index):
        # Get the range of two numbers before and after
        start_index = max(0, index - 1)  # If 0, only take 0 and 1
        end_index = min(len(data) - 1, index + 1)  # If it's the last number, only take itself and the previous one
        
        # Extract entities in specified range
        entities = []
        for i in range(start_index, end_index + 1):
            entities.extend(data.get(str(i), {}).get("entities", []))
        # Remove source_id from each entity
        for entity in entities:
            entity.pop("source_id", None)
        return entities

def get_nearby_relationships(data, index):
        # Get the range of two numbers before and after
        start_index = max(0, index - 1)  # If 0, only take 0 and 1
        end_index = min(len(data) - 1, index + 1)  # If it's the last number, only take itself and the previous one
        
        # Extract relationships in specified range
        relationships = []
        for i in range(start_index, end_index + 1):
            relationships.extend(data.get(str(i), {}).get("relationships", []))
        # Remove source_id from each relationship
        for relationship in relationships:
            relationship.pop("source_id", None)
        return relationships

def align_single_image_entity(img_entity_name, text_chunks):
    image_data = get_image_data()
    image_path = image_data[img_entity_name]["image_path"]
    img_entity_description = image_data[img_entity_name]["description"]
    chunk_order_index = image_data[img_entity_name]["chunk_order_index"]
    nearby_chunks = get_nearby_chunks(text_chunks, chunk_order_index)
    entity_type = PROMPTS["DEFAULT_ENTITY_TYPES"]
    entity_type = [item.upper() for item in entity_type]
    with open(image_path, "rb") as image_file:
        img_base = base64.b64encode(image_file.read()).decode("utf-8")
    alignment_prompt_user = PROMPTS["image_entity_alignment_user"].format(entity_type = entity_type, img_entity=img_entity_name, img_entity_description=img_entity_description, chunk_text=nearby_chunks)
    aligned_image_entity = get_mmllm_response(alignment_prompt_user, PROMPTS["image_entity_alignment_system"], img_base)
    return normalize_to_json(aligned_image_entity)

def judge_image_entity_alignment(image_entity_name, image_entity_description, possible_image_matched_entities, nearby_chunks):
    image_entity_judgement_user = PROMPTS["image_entity_judgement_user"].format(img_entity=image_entity_name, img_entity_description=image_entity_description, possible_matched_entities=possible_image_matched_entities, chunk_text=nearby_chunks)
    matched_entity_name = get_llm_response(image_entity_judgement_user, PROMPTS["image_entity_judgement_system"])
    return matched_entity_name

def get_possible_entities_image_clustering(
    image_entity_description, nearby_text_entity_list, nearby_relationship_list
):
    # Step 0: Sort relationship list by weight in descending order
    nearby_relationship_list = sorted(nearby_relationship_list, key=lambda x: x['weight'], reverse=True)
    
    # Step 1: Get embeddings of all entity descriptions
    descriptions = [entity["description"] for entity in nearby_text_entity_list]
    entity_names = [entity["entity_name"] for entity in nearby_text_entity_list]
    embeddings = encode(descriptions)

    # Step 2: Spectral clustering
    # Compute similarity matrix (cosine similarity)
    similarity_matrix = cosine_similarity(embeddings)

    # Modify degree matrix based on relationship weights
    for relation in nearby_relationship_list:
        # Only execute when both src_id and tgt_id are in entity_names
        if relation["src_id"] in entity_names and relation["tgt_id"] in entity_names:
            src_idx = entity_names.index(relation["src_id"])
            tgt_idx = entity_names.index(relation["tgt_id"])
        else:
            continue

        weight = relation["weight"]
        similarity_matrix[src_idx, tgt_idx] *= weight
        similarity_matrix[tgt_idx, src_idx] *= weight  # Ensure adjacency matrix is symmetric
    
    # Compute degree matrix
    degree_matrix = np.zeros_like(similarity_matrix)
    for i in range(len(similarity_matrix)):
        degree_matrix[i, i] = np.sum(similarity_matrix[i, :])

    # Compute Laplacian matrix L = D - A
    laplacian_matrix = degree_matrix - similarity_matrix

    # Compute eigenvalues and eigenvectors of Laplacian matrix
    eigvals, eigvecs = np.linalg.eig(laplacian_matrix)

    # Select eigenvectors corresponding to the k smallest eigenvalues
    k = max(2, math.ceil(math.sqrt(len(nearby_text_entity_list))))
    eigvecs_selected = eigvecs[:, np.argsort(eigvals)[:k]]
    # The Laplacian matrix may result in non-real symmetric matrix producing complex numbers, so take absolute value
    eigvecs_selected = np.abs(eigvecs_selected)

    # Use DBSCAN clustering
    min_samples = max(1, math.ceil(len(nearby_text_entity_list) / 10))
    dbscan = DBSCAN(eps=0.5, min_samples=min_samples)  # Adjust eps and min_samples parameters
    dbscan_labels = dbscan.fit_predict(eigvecs_selected)

    # Output clustering label for each node
    labels = dbscan_labels

    # Output labels in the order of nearby_text_entity_list
    labels = [labels[entity_names.index(entity["entity_name"])] for entity in nearby_text_entity_list]

    # Step 3: Determine category of input description
    input_embedding = encode([image_entity_description])
    # Check number of samples in training data
    n_samples_fit = embeddings.shape[0]
    # Set n_neighbors, ensuring it doesn't exceed the number of samples in training data
    n_neighbors = min(3, n_samples_fit)
    # Find nearest neighbors and use labels from Pagerank, Leiden or Spectral
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(embeddings)
    # Find nearest neighbors and use labels from Pagerank, Leiden or Spectral
    nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(embeddings)
    _, nearest_idx = nn.kneighbors(input_embedding)
    target_label = labels[nearest_idx[0][0]]

    # Step 4: Output all entity information belonging to this category
    result_entities = [
        entity
        for entity, label in zip(nearby_text_entity_list, labels)
        if label == target_label
    ]

    return result_entities

def get_possible_entities_text_clustering(
   filtered_image_entity_list, nearby_text_entity_list, nearby_relationship_list
):
    """
    Clustering and classification function supporting KMeans/DBSCAN/Pagerank/Leiden clustering and KNN/LLM classification.

    Parameters:
        clustering_method (str): Clustering method ("KMeans", "DBSCAN", "Pagerank", or "Leiden").
        classify_method (str): Classification method ("knn" or "llm").
        filtered_image_entity_list (list): Filtered image entity list, each entity contains "entity_name" and "description".
        nearby_text_entity_list (list): Nearby text entity list, each entity contains "entity_name", "entity_type" and "description".
        nearby_relationship_list (list): Relationship list between entities, each relationship contains "src_id", "tgt_id", "weight" and "description".

    Returns:
        image_entity_with_labels (list): List of image entities with corresponding categories, each item is {"entity_name": ..., "label": ..., "description": ..., "entity_type": ...}.
        text_clustering_results (list): Clustered text entity list, each item is {"label": ..., "entities": [...]}.
    """
    # Step 0: Sort relationship list by weight in descending order
    nearby_relationship_list = sorted(nearby_relationship_list, key=lambda x: x['weight'], reverse=True)

    # Step 1: Get embeddings of text entity descriptions
    descriptions = [entity["description"] for entity in nearby_text_entity_list]
    entity_names = [entity["entity_name"] for entity in nearby_text_entity_list]
    embeddings = encode(descriptions)

    # Step 2: Spectral clustering
    # Compute similarity matrix (cosine similarity)
    similarity_matrix = cosine_similarity(embeddings)

    # Modify degree matrix based on relationship weights
    for relation in nearby_relationship_list:
        # Only execute when both src_id and tgt_id are in entity_names
        if relation["src_id"] in entity_names and relation["tgt_id"] in entity_names:
            src_idx = entity_names.index(relation["src_id"])
            tgt_idx = entity_names.index(relation["tgt_id"])
        else:
            continue

        weight = relation["weight"]
        similarity_matrix[src_idx, tgt_idx] *= weight
        similarity_matrix[tgt_idx, src_idx] *= weight  # Ensure adjacency matrix is symmetric
    
    # Compute degree matrix
    degree_matrix = np.zeros_like(similarity_matrix)
    for i in range(len(similarity_matrix)):
        degree_matrix[i, i] = np.sum(similarity_matrix[i, :])

    # Compute Laplacian matrix L = D - A
    laplacian_matrix = degree_matrix - similarity_matrix

    # Compute eigenvalues and eigenvectors of Laplacian matrix
    eigvals, eigvecs = np.linalg.eig(laplacian_matrix)

    # Select eigenvectors corresponding to the k smallest eigenvalues
    k = max(2, math.ceil(math.sqrt(len(nearby_text_entity_list))))
    eigvecs_selected = eigvecs[:, np.argsort(eigvals)[:k]]
    # The Laplacian matrix may result in non-real symmetric matrix producing complex numbers, so take absolute value
    eigvecs_selected = np.abs(eigvecs_selected)

    # Use DBSCAN clustering
    min_samples = max(1, math.ceil(len(nearby_text_entity_list) / 10))
    dbscan = DBSCAN(eps=0.5, min_samples=min_samples)  # Adjust eps and min_samples parameters
    dbscan_labels = dbscan.fit_predict(eigvecs_selected)

    # Output clustering label for each node
    labels = dbscan_labels

    # Create a dictionary, initialize all entity labels to -1, indicating unclassified
    entity_labels = {entity["entity_name"]: -1 for entity in nearby_text_entity_list}

    # Assign labels according to clustering results
    for idx, entity in enumerate(nearby_text_entity_list):
        entity_labels[entity["entity_name"]] = labels[idx]

    # Output labels in the order of nearby_text_entity_list
    labels = [entity_labels[entity["entity_name"]] for entity in nearby_text_entity_list]

    # Step 3: Classify image entities into clustering categories
    image_entity_with_labels = []
    input_embeddings = encode([entity["description"] for entity in filtered_image_entity_list])
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(embeddings)
    for image_entity, input_embedding in zip(filtered_image_entity_list, input_embeddings):
        _, nearest_idx = nn.kneighbors([input_embedding])
        target_label = labels[nearest_idx[0][0]]
        image_entity_with_labels.append({
            "entity_name": image_entity["entity_name"],
            "label": target_label,
            "description": image_entity["description"],
            "entity_type": image_entity.get("entity_type", "image")
        })
    
    # Step 4: Generate clustering results
    text_clustering_results = []
    for label in set(labels):
        text_clustering_results.append({
            "label": label,
            "entities": [
                {
                    "entity_name": nearby_text_entity_list[idx]["entity_name"],
                    "entity_type": nearby_text_entity_list[idx]["entity_type"],
                    "description": nearby_text_entity_list[idx]["description"],
                }
                for idx in range(len(labels))
                if labels[idx] == label
            ]
        })

    return image_entity_with_labels, text_clustering_results

def judge_text_entity_alignment_clustering(image_entity_with_labels, text_clustering_results):
    """
    Use LLM to determine whether entity fusion is needed and output fusion results.

    Parameters:
        image_entity_with_labels (list): List of image entities with corresponding categories, each item is {"entity_name": ..., "label": ..., "description": ..., "entity_type": ...}.
        text_clustering_results (list): Clustered text entity list, each item is {"label": ..., "entities": [...]}.

    Returns:
        merged_entities (list): Fused entity list, each item is {
            "entity_name": ..., 
            "entity_type": ..., 
            "description": ..., 
            "source_image_entities": [...], 
            "source_text_entities": [...]
        }.
    """
    # Build context for fusion task
    clusters_info = []
    for cluster in text_clustering_results:
        clusters_info.append({
            "label": cluster["label"],
            "text_entities": [
                {
                    "entity_name": entity["entity_name"],
                    "entity_type": entity["entity_type"],
                    "description": entity["description"],
                }
                for entity in cluster["entities"]
            ]
        })

    # Build input prompt
    prompt_user = f"""
You are tasked with aligning image entities and text entities based on their labels and descriptions. Below are the clusters and the entities they contain.

Clusters information:
{{
    "clusters": [
        {", ".join([f'{{"label": {c["label"]}, "text_entities": {c["text_entities"]}}}' for c in clusters_info])}
    ]
}}

Image entities with labels:
{[
    {
        "entity_name": e["entity_name"],
        "label": e["label"],
        "description": e["description"],
        "entity_type": e["entity_type"]
    }
    for e in image_entity_with_labels
]}

Instruction:
1. For each image entity, look at the corresponding cluster (same label).
2. Compare the description and type of the image entity with the text entities in the same cluster.
3. Identify matching entities between the image entities and text entities within the same cluster (same label).
4. For each match, create a new unified entity by merging the descriptions and including the source entities under "source_image_entities" and "source_text_entities".
5. Output a JSON list where each item represents a merged entity with the following structure:
    {{
        "entity_name": "Newly merged entity name",
        "entity_type": "Type of the merged entity",
        "description": "Merged description of the entity",
        "source_image_entities": ["List of matched image entity names"],
        "source_text_entities": ["List of matched text entity names"]
    }}
Include only one JSON list as the output, strictly following the structure above.
"""
    prompt_system = """You are an AI assistant skilled in aligning entities based on semantic descriptions and cluster information. Use the provided instructions to merge entities accurately."""

    # Call LLM to get fusion results
    merged_entities = get_llm_response(cur_prompt=prompt_user, system_content=prompt_system)
    normalized_merged_entities = normalize_to_json_list(merged_entities)
    return [
    item for item in normalized_merged_entities 
    if item.get("source_image_entities") and item.get("source_text_entities")
]

def extract_image_entities(img_entity_name):
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    working_dir = global_config['working_dir']
    # Build GraphML file path
    image_knowledge_graph_path = os.path.join(working_dir, f"images/{img_entity_name}/graph_{img_entity_name}_entity_relation.graphml")
    
    # Check if file exists
    if not os.path.exists(image_knowledge_graph_path):
        print(f"GraphML file not found: {image_knowledge_graph_path}")
        return

    # Parse GraphML file
    tree = ET.parse(image_knowledge_graph_path)
    root = tree.getroot()
    image_entities = []
    # Define namespace
    namespace = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
    # Traverse all 'node' elements
    for node in root.findall('graphml:graph/graphml:node', namespace):
        # Extract entity information
        entity_name = node.get('id').strip('"')
        for data in node.findall('graphml:data', namespace):
            if data.get('key') == 'd0':  # 'd0' corresponds to entity type
                entity_type = data.text.strip('"')  # Get entity type and remove quotes
        for data in node.findall('graphml:data', namespace):
            if data.get('key') == 'd1':  # 'd1' corresponds to description
                description = data.text.strip('"')  # Get description and remove quotes

        # Prepare node data
        node_data = {
            "entity_name": entity_name,
            "entity_type": entity_type,
            "description": description
        }
        image_entities.append(node_data)
    return image_entities

def enhance_image_entities(enhanced_image_entity_list, nearby_chunks):
    enhance_image_entity_user = PROMPTS["enhance_image_entity_user"].format(enhanced_image_entity_list=enhanced_image_entity_list, chunk_text=nearby_chunks)
    enhanced_image_entities = get_llm_response(enhance_image_entity_user, PROMPTS["enhance_image_entity_system"])
    return normalize_to_json_list(enhanced_image_entities)

def ensure_quoted(entity_name):
    # Check if string starts and ends with double quotes
    if not (entity_name.startswith('"') and entity_name.endswith('"')):
        # If no double quotes, add them
        entity_name = f'"{entity_name}"'
    return entity_name

def image_knowledge_graph_alignment(image_entity_name):
    image_data = get_image_data()
    chunk_knowledge_graph = get_chunk_knowledge_graph()
    chunk_order_index = image_data[image_entity_name].get("chunk_order_index")
    image_entity_list = extract_image_entities(image_entity_name)
    exclude_types=["ORI_IMG", "IMG"]
    filtered_image_entity_list = [entity for entity in image_entity_list if entity['entity_type'] not in exclude_types]
    nearby_text_entity_list = get_nearby_entities(chunk_knowledge_graph, chunk_order_index)
    nearby_relationship_list = get_nearby_relationships(chunk_knowledge_graph, chunk_order_index)
    image_entity_with_labels, text_clustering_results = get_possible_entities_text_clustering(filtered_image_entity_list, nearby_text_entity_list, nearby_relationship_list)
    aligned_text_entity_list = judge_text_entity_alignment_clustering(image_entity_with_labels, text_clustering_results)
    return aligned_text_entity_list

def enhanced_image_knowledge_graph(aligned_text_entity_list, image_entity_name):
    image_data = get_image_data()
    text_chunks = get_text_chunks()
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    working_dir = global_config['working_dir']
    img_kg_path = os.path.join(working_dir, f'images/{image_entity_name}/graph_{image_entity_name}_entity_relation.graphml')
    enhanced_img_kg_path = os.path.join(working_dir, f'images/{image_entity_name}/enhanced_graph_{image_entity_name}_entity_relation.graphml')
    image_entity_list = extract_image_entities(image_entity_name)
    exclude_types=["ORI_IMG", "IMG"]
    filtered_image_entity_list = [entity for entity in image_entity_list if entity['entity_type'] not in exclude_types]
    chunk_order_index = image_data[image_entity_name]["chunk_order_index"]
    nearby_chunks = get_nearby_chunks(text_chunks, chunk_order_index)
    # Safely access 'source_image_entities' and ensure there are no errors
    source_image_entities = []
    for entity in aligned_text_entity_list:
        if 'source_image_entities' in entity and entity['source_image_entities']:
            source_image_entities.append(entity['source_image_entities'][0])
        else:
            # If 'source_image_entities' is missing or empty, we skip this entity
            print(f"Warning: 'source_image_entities' is missing or empty for entity {entity}. Skipping.")
    # Filter out image entities that have a matching 'entity_name'
    enhanced_image_entity_list = [entity for entity in filtered_image_entity_list if entity['entity_name'] not in source_image_entities]
    enhanced_image_entities = enhance_image_entities(enhanced_image_entity_list, nearby_chunks)
    
    # Step 1: Load the original knowledge graph
    G = nx.read_graphml(img_kg_path)
    
    # Step 2: Update the graph with enhanced entity details for nodes
    for entity in enhanced_image_entities:
        original_name = entity.get('original_name', None)  # Use .get() to avoid KeyError
        if original_name is None:
            print(f"Warning: 'original_name' is missing for entity {entity}. Skipping update.")
            continue  # Skip this iteration if 'original_name' is missing

        entity['entity_name'] = ensure_quoted(entity['entity_name'])

        # If 'description' is missing, skip this entity
        if 'description' not in entity:
            print(f"Warning: 'description' is missing for entity {entity}. Skipping update.")
            continue

        # Update nodes based on original name
        for node in G.nodes(data=True):
            node_id = node[0].strip('"')  # Remove extra quotes around node IDs
            if original_name == node_id:
                # Modify node_id to the new entity_name
                G = nx.relabel_nodes(G, {node[0]: entity['entity_name']})

                # Update the node's description with the enhanced entity's description
                G.nodes[entity['entity_name']].update({'description': entity['description']})
        
        # Update edges where the source or target node matches the original name
        for edge in G.edges(data=True):
            source, target, edge_data = edge
            
            if original_name == source or original_name == target:
                # Modify source_id and target_id in the edge based on the entity_name
                if original_name == source:
                    source = entity['entity_name']
                if original_name == target:
                    target = entity['entity_name']

    # Step 3: Save the updated graph to a new GraphML file
    nx.write_graphml(G, enhanced_img_kg_path)
    return enhanced_img_kg_path

def image_knowledge_graph_update(enhanced_img_kg_path, image_entity_name):
    image_data = get_image_data()
    text_chunks = get_text_chunks()
    chunk_knowledge_graph = get_chunk_knowledge_graph()
    img_kg_path = enhanced_img_kg_path
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    working_dir = global_config['working_dir']
    new_img_kg_path = os.path.join(working_dir, f'images/{image_entity_name}/new_graph_{image_entity_name}_entity_relation.graphml')
    
    image_entity = align_single_image_entity(image_entity_name, text_chunks)
    chunk_order_index = image_data[image_entity_name].get("chunk_order_index")
    nearby_chunks = get_nearby_chunks(text_chunks, chunk_order_index)
    nearby_text_entity_list = get_nearby_entities(chunk_knowledge_graph, chunk_order_index)
    nearby_relationship_list = get_nearby_relationships(chunk_knowledge_graph, chunk_order_index)
    
    
    if image_entity is not None:
        image_entity_name = image_entity.get("entity_name","no_match")
        image_entity_description = image_entity.get("description","None.")
    else:
        return img_kg_path
    
    if image_entity_name == "no_match" or image_entity_name == "no match":
        return img_kg_path
    
    possible_image_matched_entities = get_possible_entities_image_clustering(image_entity_description, nearby_text_entity_list, nearby_relationship_list) 
    matched_entity_name = judge_image_entity_alignment(image_entity_name, image_entity_description, possible_image_matched_entities, nearby_chunks)
    
    matched_entity_name_normalized = matched_entity_name.strip().replace(" ", "").replace("\\", "").lower()
    
    G = nx.read_graphml(img_kg_path)
    matched_entity_found = False  # Flag to check if matched_entity_name is found
    
    for entity in nearby_text_entity_list:
        entity_name_normalized = entity["entity_name"].strip().replace(" ", "").replace("\\", "").lower()
        
        if matched_entity_name_normalized == entity_name_normalized:
            matched_entity_found = True
            source_node_id = None
            for node, data in G.nodes(data=True):
                if data.get('entity_type') == '"ORI_IMG"' or '"UNKNOWN"':
                    source_node_id = node
                    break

            if source_node_id is None:
                raise ValueError("No node with entity_type 'ORI_IMG' found in the graph.")
            
            if len(G.edges(data=True)) > 0:
                first_edge = list(G.edges(data=True))[0]
                data_source_id_value = first_edge[2].get("source_id", "")
                data_order_value = first_edge[2].get("order", "")
            else:
                # Handle the case where there are no edges
                data_source_id_value = G.nodes[source_node_id].get('source_id', '')
                data_order_value = 1  # Set a default

            entity["entity_name"] = ensure_quoted(entity["entity_name"])
            G.add_edge(source_node_id, entity["entity_name"], 
                       weight=10.0, 
                       description=f"{source_node_id} is the image of {entity['entity_name']}.",
                       source_id=data_source_id_value,
                       order=data_order_value)
            G.add_node(entity["entity_name"], 
                        entity_type=entity["entity_type"], 
                        description=entity["description"],
                        source_id=data_source_id_value)
            
            # Save the updated graph to a new path
            nx.write_graphml(G, new_img_kg_path)
            break  # Exit the loop as we have processed the matched entity
    
    if not matched_entity_found:
        # If matched entity is not found, add a new node for image_entity_name
        source_node_id = None
        for node, data in G.nodes(data=True):
            if data.get('entity_type') == '"ORI_IMG"' or '"UNKNOWN"':
                source_node_id = node
                break
        
        if source_node_id is None:
            raise ValueError("No node with entity_type 'ORI_IMG' found in the graph.")
        
        
        # Create a relationship between the new node and the "ORI_IMG" node
        if len(G.edges(data=True)) > 0:
            first_edge = list(G.edges(data=True))[0]
            source_id_value = first_edge[2].get("source_id", "")
            order_value = first_edge[2].get("order", "")
        else:
            # Handle the case where there are no edges
            source_id_value = G.nodes[source_node_id].get('source_id', '')
            order_value = 1  # Set a default

        # Add new node with the image entity's name and description
        G.add_node(image_entity_name, 
                   entity_type="IMG_ENTITY", 
                   description=image_entity_description,
                   source_id=source_id_value)
        
        G.add_edge(source_node_id, image_entity_name, 
                   weight=10.0, 
                   description=f"{source_node_id} is the image of {image_entity_name}.",
                   source_id=source_id_value,
                   order=order_value)
        
        # Save the updated graph to a new path
        nx.write_graphml(G, new_img_kg_path)
    
    return new_img_kg_path

def merge_graphs(image_graph_path, graph_path, aligned_text_entity_list, image_entity_name):
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    working_dir = global_config['working_dir']
    merged_kg_path = os.path.join(working_dir, f'graph_merged_{image_entity_name}.graphml')

    # Step 1: Load image and text knowledge graphs
    image_graph = nx.read_graphml(image_graph_path)
    text_graph = nx.read_graphml(graph_path)
    
    # If graph loading fails, print error and return
    if image_graph is None or text_graph is None:
        print(f"Failed to load graphs, please check file paths.")
        return
    
    # Step 2: Merge two graphs
    # Use nx.compose to merge nodes and edges of two graphs
    merged_graph = nx.compose(image_graph, text_graph)
    # Step 3: Traverse aligned entity list for fusion
    for entity_info in aligned_text_entity_list:
        # Check if result has issues, if missing fields exist, skip this entity
        if not all(key in entity_info for key in ['entity_name', 'entity_type', 'description', 'source_image_entities', 'source_text_entities']):
            continue 
        entity_name = entity_info['entity_name']  # Fused entity name
        entity_type = entity_info['entity_type']  # Fused entity type
        description = entity_info['description']  # Fused entity description
        
        # Get nodes corresponding to image and text entities
        source_image_entities = entity_info['source_image_entities']
        source_text_entities = entity_info['source_text_entities']
        
        # When getting source_id from image and text graphs, ensure quotes are removed
        source_image_entity = ensure_quoted(source_image_entities[0])
        source_text_entity = ensure_quoted(source_text_entities[0])
        
        # Ensure these nodes exist in the graph
        if source_image_entity in image_graph.nodes:
            source_id_image = image_graph.nodes[source_image_entity].get('source_id', '')
        else:
            print(f"Node {source_image_entity} does not exist in image graph")
            continue

        if source_text_entity in text_graph.nodes:
            source_id_text = text_graph.nodes[source_text_entity].get('source_id', '')
        else:
            print(f"Node {source_text_entity} does not exist in text graph")
            continue
        # Connect two source_ids together
        source_id = GRAPH_FIELD_SEP.join([source_id_image, source_id_text])

        # Step 4: Fuse nodes
        # Assume source_image_entities[0] is the target node, connect all other entities to this node
        target_entity = ensure_quoted(source_image_entities[0])

        # Merge all entities in source_image_entities and source_text_entities
        all_entities = source_image_entities + source_text_entities

        # Ensure duplicate entities are removed
        all_entities = list(set(all_entities))  # Remove duplicates

        # First traverse all entities, connecting them with the target entity
        for entity in all_entities:
            entity = ensure_quoted(entity)
            if entity != target_entity and entity in merged_graph.nodes: 
                neighbors = list(merged_graph.neighbors(entity))  # Get neighbors of current entity
                for neighbor in neighbors:
                    if not merged_graph.has_edge(target_entity, neighbor):
                        merged_graph.add_edge(target_entity, neighbor)  # Connect neighbor with target entity
                    # Merge edge attributes to target node's edges
                    edge_data = merged_graph.get_edge_data(entity, neighbor)
                    target_edge_data = merged_graph.get_edge_data(target_entity, neighbor)
                    
                    # Merge edge attributes (pass all attributes such as weight, description, source_id and order)
                    if target_edge_data:
                        # If edge already exists, merge existing attributes
                        for key in edge_data:
                            if key in ['weight', 'description', 'source_id', 'order']:
                                # Merge edge attributes, add new value if exists, or keep existing one
                                target_edge_data[key] = edge_data.get(key, target_edge_data.get(key))
                    else:
                        # If edge doesn't exist, add new edge attributes
                        merged_graph[target_entity][neighbor].update(edge_data)

                merged_graph.remove_node(entity)  # Remove already merged entity node
        
        # Before updating, check if target node exists, create it if not
        if target_entity not in merged_graph.nodes:
            merged_graph.add_node(target_entity)
        # Modify target node attributes
        merged_graph.nodes[target_entity].update({
            'entity_type': entity_type,
            'description': description,
            'source_id': source_id
        })
        merged_graph = nx.relabel_nodes(merged_graph, {target_entity: ensure_quoted(entity_name)})
    
    # Step 5: Save merged graph
    # Use NetworkX to save merged graph to specified path
    nx.write_graphml(merged_graph, merged_kg_path)
    logger.info(f"Merged knowledge graph saved to: {merged_kg_path}")
    return merged_kg_path

async def fusion(img_ids):
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    working_dir = global_config['working_dir']
    graph_path = os.path.join(working_dir, 'graph_chunk_entity_relation.graphml')
    for image_entity_name in img_ids:
        merged_kg_path = os.path.join(working_dir, f'graph_merged_{image_entity_name}.graphml')
        if os.path.exists(merged_kg_path):
            continue
        aligned_text_entity_list = image_knowledge_graph_alignment(image_entity_name)
        enhanced_img_kg_path = enhanced_image_knowledge_graph(aligned_text_entity_list, image_entity_name)
        image_graph_path = image_knowledge_graph_update(enhanced_img_kg_path, image_entity_name)
        graph_path = merge_graphs(image_graph_path, graph_path, aligned_text_entity_list, image_entity_name)
    createvdb = create_EntityVDB()
    return await createvdb.create_vdb()
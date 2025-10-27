from dataclasses import dataclass, field
from typing import TypedDict, Union, Generic, TypeVar, cast, Any
from nano_vectordb import NanoVectorDB
import html
import networkx as nx
import numpy as np
import os
import asyncio

from base import (
    logger,
    EmbeddingFunc,
    load_json,
    write_json,
)

# Text chunk storage structure
TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)
@dataclass
class StorageNameSpace:
    """
    Storage namespace class for managing storage operations.

    Attributes:
    namespace (str): Namespace name.
    global_config (dict): Global configuration information.

    Methods:
    index_done_callback: Commit storage operations after indexing.
    query_done_callback: Commit storage operations after querying.
    """
    namespace: str
    global_config: dict

    async def index_done_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def query_done_callback(self):
        """commit the storage operations after querying"""
        pass

@dataclass
class BaseVectorStorage(StorageNameSpace):
    """
    Base vector storage class, inherits from StorageNameSpace.

    Attributes:
    namespace (str): Namespace name.
    global_config (Dict[str, Any]): Global configuration information.
    embedding_func (EmbeddingFunc): Vector embedding function.
    meta_fields (Set[str]): Meta fields set, defaults to empty set.

    Methods:
    query: Query method, specific function is in _storage.py, same below.
    upsert: Insert or update method.
    """
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError


# Used to define generic functions
T = TypeVar("T")

@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    """
    Base key-value storage class, inherits from StorageNameSpace.

    Attributes:
    namespace (str): Namespace name.
    global_config (Dict[str, any]): Global configuration information.

    Methods:
    all_keys: Get all keys.
    get_by_id: Get data by ID.
    get_by_ids: Get data by multiple IDs.
    filter_keys: Filter out non-existent keys.
    upsert: Insert or update data.
    drop: Delete entire storage space.
    """
    async def all_keys(self) -> list[str]:
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    async def drop(self):
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace):
    """
    Base graph storage class, inherits from StorageNameSpace.

    Attributes:
    namespace (str): Namespace name.
    global_config (Dict[str, any]): Global configuration information.

    Methods:
    has_node: Check if node exists.
    has_edge: Check if edge exists.
    node_degree: Get node degree.
    edge_degree: Get edge degree.
    get_node: Get node information.
    get_edge: Get edge information.
    get_node_edges: Get all edges of a node.
    upsert_node: Insert or update node.
    upsert_edge: Insert or update edge.
    clustering: Perform graph clustering.
    community_schema: Get community structure.
    embed_nodes: Embed nodes.
    """
    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        raise NotImplementedError

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        raise NotImplementedError

    async def clustering(self, algorithm: str):
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("Node embedding is not used in nano-graphrag.")
    
@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        # Initialize, determine working directory based on global configuration to get complete file path
        working_dir = self.global_config["working_dir"]
        # Generate specific JSON filename based on namespace for storing key-value data
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        # Load stored data, if file doesn't exist or is empty, initialize as empty dictionary
        self._data = load_json(self._file_name) or {}
        # Print log showing number of loaded data entries
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")
    # Get list of all keys
    async def all_keys(self) -> list[str]:
        return list(self._data.keys())
    # After index operation is complete, write current data to JSON file
    async def index_done_callback(self):
        write_json(self._data, self._file_name)
    # Get data by ID
    async def get_by_id(self, id):
        return self._data.get(id, None)
    
    async def get_by_ids(self, ids, fields=None):
        """
        Get data items by ID list.

        Parameters:
        ids (list): List of IDs for which to get data.
        fields (list, optional): Limit fields in returned data. If not provided, defaults to None and will return complete data items.

        Returns:
        list: List of data items arranged in order of specified ID list. If some IDs have no data items found, corresponding positions will be None.
        """
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                # If data item exists and ID is in _data dictionary, construct a new dictionary containing only fields from fields
                {k: v for k, v in self._data[id].items() if k in fields}
                # Check if _id is in dataset to avoid KeyError
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]
    # Filter out list of keys not in data storage
    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])
    # Insert or update data
    async def upsert(self, data: dict[str, dict]):
        self._data.update(data)
    # Clear current stored data
    async def drop(self):
        self._data = {}

@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    # Cosine similarity threshold determining quality of returned results
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        # Initialize vector database storage file and embedding configuration
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        # Initialize vector database client (NanoVectorDB) and set embedding dimension
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        # Get query similarity threshold from global config, or use default value
        self.cosine_better_than_threshold = self.global_config.get(
            "query_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: dict[str, dict]):
        """
        Insert or update vector data.

        This method is used to insert or update dictionary-formatted data into the vector database. The data is first 
        converted to a suitable insertion format, then processed in batches to avoid performance issues from inserting 
        too much data at once. Afterwards, embedding vectors are computed asynchronously for each batch of data, 
        and these vectors are attached to the data entries. Finally, the client's insert or update method is called to complete the operation.

        Parameters:
        data: dict[str, dict] - A dictionary where keys are unique identifiers for data, and values are dictionaries containing the actual data content.

        Returns:
        The result of the insert or update operation.
        """
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
       # Convert data to a list suitable for insertion and extract content
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        # Process data in batches
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        # Asynchronously compute embedding vectors for each batch of data
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        # Merge all batch embedding vectors into one large array
        embeddings = np.concatenate(embeddings_list)
        # Attach the computed embedding vectors to each data entry
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        # Call the client's insert or update method to complete data insertion or update
        results = self._client.upsert(datas=list_data)
        return results

    async def query(self, query: str, top_k=5):
        """
        Get the most relevant documents based on the provided query string.

        This asynchronous method uses a pre-trained embedding function to convert the query into an embedding representation,
        then searches the embedding index for documents most similar to the query.

        Parameters:
        - query: str, the user's query string.
        - top_k: int, the number of most relevant documents to return, default is 5.

        Returns:
        - A list containing the most relevant documents and their similarity distance to the query.
        """
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        # Format results, add document id and distance information
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        self._client.save()

@dataclass
class NetworkXStorage(BaseGraphStorage):
    # Load and return a NetworkX graph, stored in graphml format.
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None
    # Write NetworkX graph to graphml file.
    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        
        Parameters:
            graph (nx.Graph): The input NetworkX graph.

        Returns:
            nx.Graph: The largest connected component of the input graph, sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        
        Parameters:
        graph (nx.Graph): The input network graph.
        
        Returns:
        nx.Graph: The stabilized network graph.
        """
        # Initialize a new graph instance based on the input graph's type 
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()
        # Sort nodes to ensure consistent node addition order
        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])
        # Add sorted nodes to the new graph
        fixed_graph.add_nodes_from(sorted_nodes)
        # Store edge data in a list for subsequent processing
        edges = list(graph.edges(data=True))
        # If the graph is not directed, sort edges to ensure consistent edge order
        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]
        # Define a function to get the edge key, used for subsequent edge sorting
        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"
        # Sort edges
        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))
        # Add sorted edges to the new graph
        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        """
        Initialization function used to load graph data and initialize related properties.

        This function first determines the path to the graphml file based on the working directory 
        in the global configuration and the instance's namespace. It then attempts to load existing 
        graph data from that path. If graph data exists, it loads using NetworkXStorage and logs 
        information including the number of nodes and edges in the graph. If no graph data exists, 
        it initializes a new undirected graph. Finally, it initializes two algorithm dictionaries 
        for graph clustering algorithms and node embedding algorithms respectively.
        """
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }
    # Write the currently stored graph to GraphML file
    async def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        """
        Asynchronously check if a specified node exists in the graph. The following few functions are similar 
        and will not be annotated - they all call NetworkX functions.

        This method is primarily used to determine whether the graph structure contains a specific node. 
        It efficiently queries whether a node exists by calling the underlying graph object's has_node method.

        Parameters:
        node_id (str): The unique identifier of the node to check.

        Returns:
        bool: Returns True if the node exists in the graph, otherwise returns False.
        """
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)
    # Get the degree of the specified node.
    async def node_degree(self, node_id: str) -> int:
        # [numberchiffre]: node_id not part of graph returns `DegreeView({})` instead of 0
        return self._graph.degree(node_id) if self._graph.has_node(node_id) else 0
    # Calculate the sum of degrees of two nodes
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return (self._graph.degree(src_id) if self._graph.has_node(src_id) else 0) + (
            self._graph.degree(tgt_id) if self._graph.has_node(tgt_id) else 0
        )

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    # Perform node embedding according to the specified algorithm.
    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        """
        Asynchronous method for embedding graph structure data using the node2vec algorithm.
    
        This method uses the node2vec_embed function from the graspologic library to perform graph embedding 
        based on the internal graph structure and configuration parameters. It first calls the embedding function, 
        then extracts the node IDs from the embedding results, and returns the embedding vectors and node ID list.
        """
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids
# from objprint import op
from tqdm import tqdm
from copy import deepcopy
from itertools import product

from gbc.data import GbcGraphFull
from gbc.utils import setup_gbc_logger, load_list_from_file
from gbc.processing.data_transforms import basic_filter_and_extract


setup_gbc_logger()

gbc_graphs = load_list_from_file(
    "data/gbc/wiki/wiki_gbc_graphs.jsonl", class_type=GbcGraphFull
)


drop_composition_descendants_list = [False, True]
drop_vertex_size_kwargs_list = [
    None,
    {"min_rel_width": 0.3, "min_rel_height": 0.3},
    {"max_rel_width": 0.7, "max_rel_size": 0.8, "min_rel_size": 0.2},
]
drop_vertex_types_list = [
    None,
    ["relation"],
    ["composition", "entity"],
    ["composition", "relation"],
    ["composition", "relation", "entity"],
]
drop_caption_types_list = [
    None,
    ["composition"],
    ["detail-image"],
    ["relation", "original-image"],
    ["detail-entity", "hardcode", "bagofwords"],
]
same_level_max_bbox_overlap_ratio_list = [None, 0.1, 0.5, 1]
max_n_vertices_list = [None, 1, 5, 10]
max_depth_list = [None, 1, 3]
subgraph_extraction_mode_list = ["bfs", "dfs"]
subgraph_edge_shuffling_list = [False, True]
keep_in_edges_list = [False, True]
keep_out_edges_list = [False, True]


# Cycle through graphs for testing
graph_count = len(gbc_graphs)
graph_index = 0  # Start index for cycling through graphs

# Iterate over all parameter combinations
parameter_combinations = product(
    drop_composition_descendants_list,
    drop_vertex_size_kwargs_list,
    drop_vertex_types_list,
    drop_caption_types_list,
    same_level_max_bbox_overlap_ratio_list,
    max_n_vertices_list,
    max_depth_list,
    subgraph_extraction_mode_list,
    subgraph_edge_shuffling_list,
    keep_in_edges_list,
    keep_out_edges_list,
)

# Run tests, 115200 iterations, could take several minutes
for combination in tqdm(parameter_combinations):
    # Cycle through graphs
    original_graph = gbc_graphs[graph_index]
    graph_index = (graph_index + 1) % graph_count  # Cycle index

    # Create a deepcopy of the graph for this test
    test_graph = deepcopy(original_graph)

    # Unpack parameters
    (
        drop_composition_descendants,
        drop_vertex_size_kwargs,
        drop_vertex_types,
        drop_caption_types,
        same_level_max_bbox_overlap_ratio,
        max_n_vertices,
        max_depth,
        subgraph_extraction_mode,
        subgraph_edge_shuffling,
        keep_in_edges,
        keep_out_edges,
    ) = combination

    # # Print parameters
    # print("Running test with parameters:")
    # print(f"  drop_composition_descendants: {drop_composition_descendants}")
    # print(f"  drop_vertex_size_kwargs: {drop_vertex_size_kwargs}")
    # print(f"  drop_vertex_types: {drop_vertex_types}")
    # print(f"  drop_caption_types: {drop_caption_types}")
    # print(f"  same_level_max_bbox_overlap_ratio: {same_level_max_bbox_overlap_ratio}")
    # print(f"  max_n_vertices: {max_n_vertices}")
    # print(f"  max_depth: {max_depth}")
    # print(f"  subgraph_extraction_mode: {subgraph_extraction_mode}")
    # print(f"  subgraph_edge_shuffling: {subgraph_edge_shuffling}")
    # print(f"  keep_in_edges: {keep_in_edges}")
    # print(f"  keep_out_edges: {keep_out_edges}")

    # Apply the function
    result_graph = basic_filter_and_extract(
        gbc_graph=test_graph,
        drop_composition_descendants=drop_composition_descendants,
        drop_vertex_size_kwargs=drop_vertex_size_kwargs,
        drop_vertex_types=drop_vertex_types,
        drop_caption_types=drop_caption_types,
        same_level_max_bbox_overlap_ratio=same_level_max_bbox_overlap_ratio,
        max_n_vertices=max_n_vertices,
        max_depth=max_depth,
        subgraph_extraction_mode=subgraph_extraction_mode,
        subraph_edge_shuffling=subgraph_edge_shuffling,
        keep_in_edges=keep_in_edges,
        keep_out_edges=keep_out_edges,
    )

    # # Log the result
    # op(result_graph.model_dump(), indent=2)  # Pretty-print the result graph

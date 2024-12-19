# from objprint import op
from tqdm import tqdm
from copy import deepcopy
from itertools import product

from gbc.data import GbcGraphFull
from gbc.utils import setup_gbc_logger, load_list_from_file
from gbc.processing.data_transforms import gbc_clip_filter


setup_gbc_logger()

gbc_graphs = load_list_from_file(
    "data/gbc/wiki/with_clip_scores/wiki_gbc_graphs_with_clip.jsonl",
    class_type=GbcGraphFull,
)


split_rather_than_filter_list = [False, True]
exclude_labels_list = [
    ["hardcode"],
    ["relation", "image-detail"],
    ["hardcode", "composition"],
]
random_filtering_probs_list = [dict(), {"short": 0.3, "detail": 0.2}]
add_bag_of_words_list = [False, True]
remove_edge_with_no_corr_list = [False, True]
filter_image_using_short_clip_scores_list = [False, True]
filter_after_selection_list = [False, True]
max_n_vertices_per_graph_list = [None, 10]
node_selection_mode_list = ["bfs", "dfs", "random"]
max_n_vertices_list = [None, 100]

# Cycle through graphs for testing
graph_count = len(gbc_graphs)
graph_index = 0  # Start index for cycling through graphs

# Iterate over all parameter combinations
parameter_combinations = product(
    split_rather_than_filter_list,
    exclude_labels_list,
    random_filtering_probs_list,
    add_bag_of_words_list,
    remove_edge_with_no_corr_list,
    filter_image_using_short_clip_scores_list,
    filter_after_selection_list,
    max_n_vertices_per_graph_list,
    node_selection_mode_list,
    max_n_vertices_list,
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
        split_rather_than_filter,
        exclude_labels,
        random_filtering_probs,
        add_bag_of_words,
        remove_edge_with_no_corr,
        filter_image_using_short_clip_scores,
        filter_after_selection,
        max_n_vertices_per_graph,
        node_selection_mode,
        max_n_vertices,
    ) = combination

    # Apply the function
    result_graph = gbc_clip_filter(
        test_graph,
        split_rather_than_filter=split_rather_than_filter,
        exclude_labels=exclude_labels,
        random_filtering_probs=random_filtering_probs,
        add_bag_of_words=add_bag_of_words,
        remove_edge_with_no_corr=remove_edge_with_no_corr,
        filter_image_using_short_clip_scores=filter_image_using_short_clip_scores,
        filter_after_selection=filter_after_selection,
        max_n_vertices_per_graph=max_n_vertices_per_graph,
        node_selection_mode=node_selection_mode,
        max_n_vertices=max_n_vertices,
    )

    # # Log the result
    # op(result_graph.model_dump(), indent=2)  # Pretty-print the result graph

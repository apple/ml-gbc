from objprint import op
from copy import deepcopy
from itertools import product

from gbc.data import GbcGraphFull
from gbc.utils import setup_gbc_logger, load_list_from_file
from gbc.processing.data_transforms import gbc_graph_to_text_and_image


setup_gbc_logger()

gbc_graphs = load_list_from_file(
    "data/gbc/wiki/wiki_gbc_graphs.jsonl", class_type=GbcGraphFull
)


text_format_list = ["set", "set_with_bbox", "concat", "structured"]
graph_traversal_mode_list = ["bfs", "dfs", "random", "topological"]
read_image_list = [False, True]
remove_repeated_suffix_list = [False, True]


# Cycle through graphs for testing
graph_count = len(gbc_graphs)
graph_index = 0  # Start index for cycling through graphs

# Iterate over all parameter combinations
parameter_combinations = product(
    text_format_list,
    graph_traversal_mode_list,
    read_image_list,
    remove_repeated_suffix_list,
)

# Run tests, 115200 iterations, could take several minutes
for combination in parameter_combinations:
    # Cycle through graphs
    original_graph = gbc_graphs[graph_index]
    graph_index = (graph_index + 1) % graph_count  # Cycle index

    # Create a deepcopy of the graph for this test
    test_graph = deepcopy(original_graph)

    # Unpack parameters
    text_format, graph_traversal_mode, read_image, remove_repeated_suffix = combination

    # Print parameters
    print("-------------------------------------------------------")
    print("Running test with parameters:")
    print(f"  text_format: {text_format}")
    print(f"  graph_traversal_mode: {graph_traversal_mode}")
    print(f"  read_image: {read_image}")
    print(f"  remove_repeated_suffix: {remove_repeated_suffix}")

    print("")

    if text_format == "structured":
        caption_agg_mode_for_structured_list = ["first", "concat"]

        for caption_agg_mode_for_structured in caption_agg_mode_for_structured_list:
            print(
                "  caption_agg_mode_for_structured: ",
                caption_agg_mode_for_structured,
            )
            print("")
            result = gbc_graph_to_text_and_image(
                gbc_graph=test_graph,
                graph_traversal_mode=graph_traversal_mode,
                text_format=text_format,
                read_image=read_image,
                remove_repeated_suffix=remove_repeated_suffix,
                caption_agg_mode_for_structured=caption_agg_mode_for_structured,
            )
            op(result, indent=2)

    else:
        result = gbc_graph_to_text_and_image(
            gbc_graph=test_graph,
            graph_traversal_mode=graph_traversal_mode,
            text_format=text_format,
            read_image=read_image,
            remove_repeated_suffix=remove_repeated_suffix,
        )
        op(result, indent=2)

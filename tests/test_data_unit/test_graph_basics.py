from objprint import op
from copy import deepcopy

from gbc.data import GbcGraphFull
from gbc.utils import setup_gbc_logger, load_list_from_file


setup_gbc_logger()

gbc_graphs = load_list_from_file(
    "data/gbc/wiki/wiki_gbc_graphs.jsonl", class_type=GbcGraphFull
)

print("The first gbc graph is:")
op(gbc_graphs[0].model_dump())


for mode in ["dfs", "bfs", "random"]:
    print("--------------------------------------------------------------")
    print(f"Extracted gbc subgraph with 5 vertices in {mode} order:")
    op(gbc_graphs[0].get_subgraph(5, mode=mode).model_dump())

gbc_graph = deepcopy(gbc_graphs[0])

gbc_graph = gbc_graph.drop_vertices_by_size(min_rel_width=0.2, min_rel_height=0.2)
print("--------------------------------------------------------------")
print("Dropping vertices with width < 0.2 or height < 0.2:")
op(gbc_graph.model_dump())


gbc_graph = gbc_graph.drop_captions_by_type(["hardcode", "composition"])
print("--------------------------------------------------------------")
print("Dropping captions of type 'hardcode' and 'composition':")
op(gbc_graph.model_dump())

gbc_graph = gbc_graph.drop_vertices_by_type(["relation"])
print("--------------------------------------------------------------")
print("Dropping vertices of type 'relation':")
op(gbc_graph.model_dump())

gbc_graph = gbc_graph.drop_vertices_by_type(["composition"])
print("--------------------------------------------------------------")
print("Dropping vertices of type 'composition':")
op(gbc_graph.model_dump())


gbc_graph = deepcopy(gbc_graphs[0])

gbc_graph = gbc_graph.drop_composition_descendants()
print("--------------------------------------------------------------")
print("Dropping composition descendants:")
op(gbc_graph.model_dump())

gbc_graph = gbc_graph.drop_vertices_by_overlap_area(
    max_overlap_ratio=0.5, keep_in_edges=False
)
print("--------------------------------------------------------------")
print("Dropping vertices so that overlap ratio < 0.5:")
op(gbc_graph.model_dump())

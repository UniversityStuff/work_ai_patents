import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData
import matplotlib.colors as mcolors

class HeteroDataVisualizer:
    def __init__(self, hetero_data: HeteroData):
        self.hetero_data = hetero_data

    def visualize(self):
        G = nx.Graph()
        

        # Add nodes
        node_colors = []
        category_color_map = {}
        color_palette = list(mcolors.TABLEAU_COLORS.values())
        color_index = 0
        
        node_counter = 0
        for node_type in self.hetero_data.node_types:
            num_nodes = self.hetero_data[node_type].num_nodes
            if node_type == 'patents':
                lens_ids = self.hetero_data[node_type].lens_ids
                categories = self.hetero_data[node_type].x.argmax(dim=1).tolist()
                for i in range(num_nodes):
                    category = categories[i]
                    if category not in category_color_map:
                        category_color_map[category] = color_palette[color_index % len(color_palette)]
                        color_index += 1
                    G.add_node(node_counter, node_type=node_type, label=lens_ids[i])
                    node_colors.append(category_color_map[category])
                    node_counter += 1
            else:
                for i in range(num_nodes):
                    G.add_node(node_counter, node_type=node_type)
                    node_colors.append('gray')  # Default color for other node types
                    node_counter += 1


        # Add edges
        for edge_type in self.hetero_data.edge_types:
            edge_index = self.hetero_data[edge_type].edge_index
            src, dst = edge_index
            edges = zip(src.tolist(), dst.tolist())
            G.add_edges_from(edges, edge_type=edge_type)
        # Draw the graph
        
        print("Start drawing")
        print("if it gets stuck, try reducing the number of nodes")
        
        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 12))
        labels = nx.get_node_attributes(G, 'label')
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=500, node_color=node_colors, font_size=5, font_weight="bold")
        # nx.draw_networkx_edge_labels(G, pos)
        plt.title("Heterogeneous Graph Visualization")
        plt.show()

# Example usage:
# processor = HeteroDataProcessor("example_table")
# processor.load_data(df_combined)
# hetero_data = processor.get_data()
# visualizer = HeteroDataVisualizer(hetero_data)
# visualizer.visualize()

# IMPORTANT Information:
# it gets really slow when the number of nodes is high
# I advice using df.sample(n=500) to reduce the number of nodes for testing
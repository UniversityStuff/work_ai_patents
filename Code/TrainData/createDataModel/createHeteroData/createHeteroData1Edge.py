import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected, RandomLinkSplit
from tqdm import tqdm

class HeteroData1Edge:
    def __init__(self):
        self.data = HeteroData()

    def load_data(self, df_combined):
        # Load nodes
        patent_mapping = {idx: i for i, idx in enumerate(df_combined["Lens ID"].unique())}
        category_mapping = {idx: i for i, idx in enumerate(df_combined["Category"].unique())}

        self.data['patents'].num_nodes = len(patent_mapping)
        self.data['category'].num_nodes = len(category_mapping)
        self.data['patents'].lens_ids = [None] * len(patent_mapping)
        for lens_id, idx in patent_mapping.items():
            self.data['patents'].lens_ids[idx] = lens_id
        print("Loaded nodes done")
        print("Length of patent_mapping: ", len(patent_mapping))
        print("Length of category_mapping: ", len(category_mapping))

        # Add category as a feature to patents
        patent_categories = torch.zeros((len(patent_mapping), len(category_mapping)))
        patent_labels = torch.zeros(len(patent_mapping), dtype=torch.long)
        for _, row in df_combined.iterrows():
            patent_id = patent_mapping[row["Lens ID"]]
            category_id = category_mapping[row["Category"]]
            patent_categories[patent_id, category_id] = 1
            patent_labels[patent_id] = category_id

        self.data['patents'].x = patent_categories
        self.data['patents'].y = patent_labels
        print("Added category features and labels to patents")

        # Create edges based on classifications
        classification_edges = {}
        for _, row in tqdm(df_combined.iterrows()):
            patent_id = patent_mapping[row["Lens ID"]]
            classifications = row["Classifications"].split("|")
            for classification in classifications:
                if classification not in classification_edges:
                    classification_edges[classification] = []
                classification_edges[classification].append(patent_id)
        print("Created classification_edges")

        patent_classification_edge_index = []
        for classification, patents in tqdm(classification_edges.items()):
            for i in range(len(patents)):
                for j in range(i + 1, len(patents)):
                    patent_classification_edge_index.append([patents[i], patents[j]])
                    patent_classification_edge_index.append([patents[j], patents[i]])

        patent_classification_edge_index = torch.tensor(patent_classification_edge_index).t().contiguous()
        self.data['patents', 'same_classification', 'patents'].edge_index = patent_classification_edge_index
        print("Created patent_classification_edge_index")
        print(patent_classification_edge_index)

        # Add a reverse relation for message passing
        self.data = ToUndirected()(self.data)

    def split_data(self):
        # Perform a link-level split into training, validation, and test edges
        transform = RandomLinkSplit(
            num_val=0.05,
            num_test=0.1,
            neg_sampling_ratio=0.0,
            edge_types=[('patents', 'same_classification', 'patents')],
            rev_edge_types=[('patents', 'same_classification_rev', 'patents')]
        )
        train_data, val_data, test_data = transform(self.data)

        return train_data, val_data, test_data
    
    def get_data(self):
        return self.data
    
# Example usage:
# processor = HeteroData1Edge1FeatureProcessor()
# processor.load_data(df_combined)
# train_data, val_data, test_data = processor.split_data()
# data = processor.get_data()
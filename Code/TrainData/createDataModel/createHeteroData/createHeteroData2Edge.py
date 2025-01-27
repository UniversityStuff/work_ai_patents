import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected, RandomLinkSplit
from tqdm import tqdm



def get_classification_levels(classification):
    # Extract hierarchical levels from classification
    # Example H04L9/0891 -> ['H', 'H04', 'H04L', 'H04L/0891']
    
    if classification == '':
        return []
    
    levels = []
    parts = classification.split('/')
    main_part = parts[0]
    sub_part = parts[1] if len(parts) > 1 else ''
    
    # Add hierarchical levels
    # levels.append(main_part[0])  # H
    # if len(main_part) > 1:
    #     levels.append(main_part[:3])  # H04
    # if len(main_part) > 3:
    #     levels.append(main_part[:4])  # H04L
    if sub_part:
        levels.append(main_part + '/' + sub_part)  # H04L/0891
    
    return levels


class HeteroData2Edge:
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


        # Create edges based on keywords
        keyword_edges = {}
        for _, row in tqdm(df_combined.iterrows()):
            patent_id = patent_mapping[row["Lens ID"]]
            keyphrases = row["Keyphrases"].split("|")
            for keyphrase in keyphrases:
                if keyphrase not in keyword_edges:
                    keyword_edges[keyphrase] = []
                keyword_edges[keyphrase].append(patent_id)
        print("Created keyphase_edges")

        patent_kephrase_edge_index = []
        for keyphrase, patents in tqdm(keyword_edges.items()):
            for i in range(len(patents)):
                for j in range(i + 1, len(patents)):
                    patent_kephrase_edge_index.append([patents[i], patents[j]])
                    patent_kephrase_edge_index.append([patents[j], patents[i]])
        
        patent_kephrase_edge_index = torch.tensor(patent_kephrase_edge_index).t().contiguous()
        self.data['patents', 'same_keyphrase', 'patents'].edge_index = patent_kephrase_edge_index
        print("Created keyphrase_index")

        # Create edges based on classifications
        classification_edges = {}
        for _, row in tqdm(df_combined.iterrows()):
            patent_id = patent_mapping[row["Lens ID"]]
            classifications = row["Classifications"].split("|")
            for classification in classifications:
                # if classification not in classification_edges:
                #     classification_edges[classification] = []
                # classification_edges[classification].append(patent_id)
                
                levels = get_classification_levels(classification)
                for level in levels:
                    if level not in classification_edges:
                        classification_edges[level] = []
                    classification_edges[level].append(patent_id)
        print("Created classification_edges")
        print("Number of classification_edges: ", len(classification_edges))

        patent_classification_edge_index = []
        for classification, patents in tqdm(classification_edges.items()):
            for i in range(len(patents)):
                for j in range(i + 1, len(patents)):
                    patent_classification_edge_index.append([patents[i], patents[j]])
                    patent_classification_edge_index.append([patents[j], patents[i]])
                    
        patent_classification_edge_index = torch.tensor(patent_classification_edge_index).t().contiguous()
        self.data['patents', 'same_classification', 'patents'].edge_index = patent_classification_edge_index
        print("Created patent_classification_edge_index")

        # Add a reverse relation for message passing
        self.data = ToUndirected()(self.data)
        
        # Add masks to data, if test_data is True. This value is set in the balancing data step if activated.
        if 'test_data' in df_combined.columns:
            print("Adding masks to data based on test_data")
            # Create train_mask and test_mask based on df_combined['test_data']
            num_nodes = self.data['patents'].num_nodes
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            for idx, row in df_combined.iterrows():
                patent_id = patent_mapping[row["Lens ID"]]
                if row['test_data']:
                    test_mask[patent_id] = True
                else:
                    train_mask[patent_id] = True

            # Add masks to data
            self.data['patents'].train_mask = train_mask
            self.data['patents'].test_mask = test_mask
            
            print(f"Number of training nodes: {train_mask.sum()}")
            print(f"Number of test nodes: {test_mask.sum()}")
    
    def get_data(self):
        return self.data
    
# Example usage:
# processor = HeteroData1Edge1FeatureProcessor()
# processor.load_data(df_combined)
# train_data, val_data, test_data = processor.split_data()
# data = processor.get_data()
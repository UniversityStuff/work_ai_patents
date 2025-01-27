import torch
from torch_geometric.data import InMemoryDataset, HeteroData


class PatentsInMemoryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        """
        root = Folder where dataset should be stored,
        split into raw_dir and processed_dir
        """
        super(PatentsInMemoryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """
        if this file exists in raw_dir, the download is not triggered.
        But the download is also skipped by default.
        """
        return "data.pt"

    @property
    def processed_file_names(self):
        """
        IMPORTANT: This file is created by the process method. Currently it is NOT possible to NOT skip the process step.
        """
        return "not_implemented.pt"
    
    def download(self):
        print("Download Step, should not reach")
        pass # Skip download sep if files don't exist, because we have no download


    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        data = torch.load(self.raw_paths[0])
        
        
        try: 
            data['patents'].train_mask
        except AttributeError:
            print("Adding masks to data based on nothing")
            # Create train_mask and test_mask
            num_nodes = data['patents'].num_nodes
            train_size = int(0.8 * num_nodes)
            # val_size = int(0.1 * num_nodes)
            val_size = 0
            test_size = num_nodes - train_size - val_size

            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

            train_mask[:train_size] = True
            val_mask[train_size:train_size + val_size] = True
            test_mask[train_size + val_size:] = True
            
            # Add masks to data
            data['patents'].train_mask = train_mask
            data['patents'].val_mask = val_mask
            data['patents'].test_mask = test_mask
        
        data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
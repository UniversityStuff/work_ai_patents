import torch
from torch_geometric.data import InMemoryDataset, HeteroData

class MyHeteroDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyHeteroDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `HeteroData` list.
        data_list = []

        # Example of creating a HeteroData object
        data = HeteroData()
        data['paper'].x = torch.randn((100, 16))
        data['author'].x = torch.randn((200, 16))
        data['paper', 'written_by', 'author'].edge_index = torch.randint(0, 100, (2, 500))

        data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx):
        data = self.data.__class__()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            data[key] = item[slices[idx]:slices[idx + 1]]
        return data
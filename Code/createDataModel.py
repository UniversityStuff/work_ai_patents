import torch
import duckdb as db
import pandas as pd

from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
con = db.connect("Data/textile_patents.duckdb")


# Possible datasets
# antiSeed
# metalTextiles
# naturalTextiles
# mineralTextiles
# syntheticTextiles
# --------------------------------- CREATE TABLES --------------------------------- #
tables = ["antiSeed", "metalTextiles", "naturalTextiles", "mineralTextiles", "syntheticTextiles"]

# # Create new combined table
# con.execute("CREATE TABLE combinedTable AS SELECT * FROM " + tables[0])
# for table in tables[1:]:
#     con.execute("INSERT INTO combinedTable SELECT * FROM " + table)
# # Create new category table
# con.execute("CREATE TABLE category AS SELECT \"Lens ID\", \"Category\" FROM combinedTable")

# --------------------------------- DATAFRAME --------------------------------- #
df_combined = con.execute("SELECT * FROM combinedTable").fetch_df()

# Clean dataframe
df_combined.drop(columns=["Combined", "Category"], inplace=True)

# The following step is needed to get both Keyphrases and Classificaitons both as features in the Data object
# Replace all ";;" values with "|" in the Classifications column
df_combined["Classifications"] = df_combined["Classifications"].str.replace(";;", "|")
# Combine Classifications and Keyphrases into one column with the "|" separator
df_combined["Classifications"] = df_combined["Classifications"] + "|" + df_combined["Keyphrases"]
df_combined.drop(columns=["Keyphrases"], inplace=True)
print(df_combined.head())

df_category = con.execute("SELECT * FROM category").fetch_df()
print(df_category.head())


# --------------------------------- LOADERS --------------------------------- #
def load_node(df,index_col,  encoders=None):
    mapping = {index: i for i, index in enumerate(df[index_col])}
    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping

def load_edge(df, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None):

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr

# --------------------------------- ENCODERS --------------------------------- #
class SeperatedBySeperatorGenericEncoder:
    def __init__(self, seperator):
        self.seperator = seperator
    
    def __call__(self, df):
        keyWords = {g for col in df.values for g in col.split(self.seperator)}
        mapping = {keyWord: i for i, keyWord in enumerate(keyWords)}
        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for keyWord in col.split(self.seperator):
                x[i, mapping[keyWord]] = 1
        return x

class IdentityEncoder:
    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)





# --------------------------------- NODES --------------------------------- #
category_x, category_mapping = load_node(df_category, index_col='Category')

patent_x, patent_mapping = load_node(
    df_combined, index_col='Lens ID', encoders={
        'Classifications': SeperatedBySeperatorGenericEncoder("|")
    })

# --------------------------------- EDGES --------------------------------- #
edge_index, edge_label = load_edge(
    df_combined,
    dst_index_col='Category',
    dst_mapping=category_mapping,
    src_index_col='Lens ID',
    src_mapping=patent_mapping,
    encoders={'rating': IdentityEncoder(dtype=torch.long)}
)






data = HeteroData()
data['patents'].x = ... # [num_patents, num_features]
data['category'].x = ... # [num_labels, num_features_category]

data['category', 'contains', 'patents'].edge_index = ... # [2, num_edges]


# 1. Add a reverse relation for message passing.
data = ToUndirected()(data)

# 2. Perform a link-level split into training, validation, and test edges.
transform = RandomLinkSplit(
    num_val=0.05,
    num_test=0.1,
    neg_sampling_ratio=0.0,
    edge_types=[('category', 'contains', 'patents')],
    rev_edge_types=[('patents', 'contains_rev', 'category')]
)
train_data, val_data, test_data = transform(data)
print(train_data)
print(val_data)
print(test_data)
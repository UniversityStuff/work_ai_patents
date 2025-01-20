import pandas as pd
import torch
import duckdb as db

from createDataModel.createHeteroData2Edges import HeteroData2EdgesProcessor
from createDataModel.createHeteroData1Edge1Feature import HeteroData1Edge1FeatureProcessor
from visualize.visualizeHeteroData import HeteroDataVisualizer

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



con = db.connect("Data/textile_patents.duckdb")

df_combined = con.execute("SELECT * FROM combinedTable").fetch_df()
df_combined.drop(columns=["Combined"], inplace=True)

# The following step is needed to get both Keyphrases and Classificaitons both as features in the Data object
# Replace all ";;" values with "|" in the Classifications column
df_combined["Classifications"] = df_combined["Classifications"].str.replace(";;", "|")

# Combine Classifications and Keyphrases into one column with the "|" separator
df_combined["Classifications"] = df_combined["Classifications"] + "|" + df_combined["Keyphrases"]

df_combined.drop(columns=["Keyphrases"], inplace=True)

# processor = HeteroData2EdgesProcessor()
processor = HeteroData1Edge1FeatureProcessor()

# To reduce the number of rows for testing
df_combined = df_combined.sample(n=80)

processor.load_data(df_combined)
data = processor.get_data()
print(data)

visualizer = HeteroDataVisualizer(data)
visualizer.visualize()

# train_data, val_data, test_data = processor.split_data()

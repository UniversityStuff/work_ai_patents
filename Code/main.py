import torch
from TrainData.createDataModel.createHeteroData.createHeteroData1Edge1Feature import HeteroData1Edge1FeatureProcessor
from TrainData.createDataModel.createInMemoryDataset.createInMemoryDataset import PatentsInMemoryDataset
from TrainData.models import HeteroGCN
from PrepareData.prepareDataForDB import prepare_data_for_db
import duckdb as db

from TrainData.trainModel import ModelTrainer
from TrainData.trainingData import TrainingData


con = db.connect("Data/textile_patents.duckdb")

# --------------------------------- Prepare Data in DB --------------------------------- #
# # All the file names in the CSVBaseData folder
# # If a file is named "test.csv" the string should be "test"
# file_names = ["Anti-Seed-3k","Set-Mineral-Textile-Without-Metal", "Set-Metal-Textiles", "Set-Natural-Textiles", "Set-Synthetic-Textiles"]
# # The table names in the database
# # Different because these will be the labels for the classification later
# table_names = ["antiSeed", "mineralTextiles", "metalTextiles", "naturalTextiles", "syntheticTextiles"]

# prepare_data_for_db(file_names, table_names, con)


# --------------------------------- Prepare Data for Training --------------------------------- #

# preprocessor = HeteroData2EdgesProcessor()
preprocessor = HeteroData1Edge1FeatureProcessor()

training_data = TrainingData(preprocessor)

training_data.create_combined_tables()
training_data.create_balanced_data()
training_data.create_hetero_data()

# Manually remove training_data from memory because it is very inefficient
del training_data

# --------------------------------- TRAIN MODEL --------------------------------- #

# Load the dataset
dataset = PatentsInMemoryDataset(root="Data/TextilePatents")
data = dataset[0]

# Print some statistics about the graph
print(data)
print(f'Number of nodes: {data["patents"].num_nodes}')
print(f'Number of edges: {data["patents", "same_classification", "patents"].edge_index.size(1)}')
print(f'Number of training nodes: {data["patents"].train_mask.sum()}')
print(f'Training node label rate: {int(data["patents"].train_mask.sum()) / data["patents"].num_nodes:.2f}')
print(f'Is undirected: {data.is_undirected()}')

print(data['patents'].x.shape)
print(data['patents'].y.shape)
print(data['patents', 'same_classification', 'patents'].edge_index.shape)
print(data['patents'].train_mask.shape)

hidden_channels = 16
learning_rate = 0.01
decay = 5e-4
epochs = 200

# Initialize model
model = HeteroGCN(hidden_channels=hidden_channels, data=data)

# Use CPU
device = torch.device("cpu")
model = model.to(device)
data = data.to(device)

# Initialize Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
criterion = torch.nn.CrossEntropyLoss()

# Train the model
model_trainer = ModelTrainer(model, data, optimizer, criterion)

model_trainer.train_model(epochs)
model_trainer.test_model()
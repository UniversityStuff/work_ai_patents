# Ideas
- train keyphrase extractor with cpc classes
- Multilevel Graph (the CPC classes are related to each other as well in a tree structure)
- use chatgbt as classifier

# Known Issues
- We have Empty CPC classes, we need to remove them (E.g. "CPC Class: """ --> This makes it very akward to work with the data, but in theory it should work)     

# Information about the datasets

- "Set Synthetic Textile" --> Highest proportion of foud by CPC classes

# Doing


# DBs
antiSeed
metalTextiles
naturalTextiles
mineralTextiles
syntheticTextiles


# Results
Datamodel:
- HeteroData1Edge1Feature.py

Variables:
- hidden_channels = 16
- learning_rate = 0.01
- decay = 5e-4
- epochs = 200

Test Accuracy internal Test:
Without balancing:
- 0.0468
With balancing and balance_limit = 1000:
- 0.2838

Datamodel:
- HeteroData1Edge1FeatureEnhanced1.py

Variables:
- hidden_channels = 16
- learning_rate = 0.01
- decay = 5e-4
- epochs = 200

Test Accuracy internal Test (with balancing):
- 0.3003


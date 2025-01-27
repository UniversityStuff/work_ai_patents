import pandas as pd
import torch
from .visualize.visualizeHeteroData import HeteroDataVisualizer
import os
import shutil
from sklearn.utils import resample



def balance_data(df_combined, balance_limit):
    """
    Balance the data where all labels are equaly represented.
    The data will be upscaled to the label with the most samples.
    
    args:
    df_combined: pd.DataFrame
        The combined data from the database.
    balance_limit: int
        The limit of samples per label. If the number of samples is higher than the limit the data will be downsampled.
    """
    
    # Shuffle the data
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate the split index
    split_index = int(0.8 * len(df_combined))

    # Create a new column 'test_data' to mark training and test data
    df_combined['test_data'] = [False] * split_index + [True] * (len(df_combined) - split_index)
    
    df_test = df_combined[df_combined['test_data'] == True]
    # Use only the part of the df_combined with ['test_data'] == False
    df_combined = df_combined[df_combined['test_data'] == False]
    
    # If balance_limit is set, downsample the data to the balance_limit
    if balance_limit is not None:
        df_combined = df_combined.groupby('Category').apply(lambda x: x.sample(min(len(x), balance_limit))).reset_index(drop=True)

    # Find the maximum number of samples in any label
    max_samples = df_combined['Category'].value_counts().max()

    # Create an empty DataFrame to store the balanced data
    df_balanced = pd.DataFrame()

    # Resample each label to have the same number of samples as the label with the most samples
    for label in df_combined['Category'].unique():
        df_label = df_combined[df_combined['Category'] == label]
        df_label_balanced = resample(df_label, replace=True, n_samples=max_samples, random_state=42)
        df_balanced = pd.concat([df_balanced, df_label_balanced])

    df_balanced.reset_index(drop=True, inplace=True)
    
    # Make "Lens ID" unique by appending a unique identifier
    df_balanced['Lens ID'] = df_balanced['Lens ID'] + "_" + df_balanced.groupby('Lens ID').cumcount().astype(str)

    # Concatenate the training and test sets back together
    df_balanced = pd.concat([df_balanced, df_test]).reset_index(drop=True)

    print("Balancing Done")
    print("Number of samples per label:")
    print(df_balanced['Category'].value_counts())
    print("Number of training samples per label:")
    print(df_balanced[df_balanced['test_data'] == False]['Category'].value_counts())
    print("Number of test samples per label:")
    print(df_balanced[df_balanced['test_data'] == True]['Category'].value_counts())
    
    return df_balanced


class TrainingData:
    """
    A Class that contains all important data for training and usefull methods 
    """
    def __init__(self, con, tables, data_model_preprocessor):
        self.con = con
        self.tables = tables
        self.data_model_preprocessor = data_model_preprocessor
    
    def create_combined_tables(self):
        """
        
        """
        # Check if combinedTable or category table exists and drop it they do
        self.con.execute("DROP TABLE IF EXISTS combinedTable")
        self.con.execute("DROP TABLE IF EXISTS category")
        
        # # # Create new combined table
        self.con.execute("CREATE TABLE combinedTable AS SELECT * FROM " + self.tables[0])
        for table in self.tables[1:]:
            self.con.execute("INSERT INTO combinedTable SELECT * FROM " + table)
        # Create new category table
        self.con.execute("CREATE TABLE category AS SELECT \"Lens ID\", \"Category\" FROM combinedTable")
    
    
    def create_balanced_combined_data(self, sample=None, balanced=True, balance_limit=1000):
        """
        Balanced data where all labels are equaly represented.
        Here the Kephraes and Classifications are combined into one column.
        
        args:
        sample: int, default=None
            Number of rows to sample from the data. Needed for testing or visualisation purposes.
        balanced: bool, default=True
            If True the data will be balanced.
        balance_limit: int, default=1000
            The limit of samples per label. If the number of samples is higher than the limit the data will be downsampled.
        """
        df_combined = self.con.execute("SELECT * FROM combinedTable").fetch_df()
        df_combined.drop(columns=["Combined"], inplace=True)

        # The following step is needed to get both Keyphrases and Classificaitons both as features in the Data object
        # Replace all ";;" values with "|" in the Classifications column
        df_combined["Classifications"] = df_combined["Classifications"].str.replace(";;", "|")

        # Combine Classifications and Keyphrases into one column with the "|" separator
        df_combined["Classifications"] = df_combined["Classifications"] + "|" + df_combined["Keyphrases"]

        df_combined.drop(columns=["Keyphrases"], inplace=True)
        
        # To reduce the number of rows for testing
        if sample is not None:
            df_combined = df_combined.sample(sample)
            
        if balanced:
            df_combined = balance_data(df_combined, balance_limit)
        
        self.df_combined = df_combined
            
        return df_combined
    
    def create_balanced_split_data(self, sample=None, balanced=True, balance_limit=1000):
        """
        Balanced data where all labels are equaly represented.
        Keyphrases and Classifications are kept as separate columns.
        
        args:
        sample: int, default=None
            Number of rows to sample from the data. Needed for testing or visualisation purposes.
        balanced: bool, default=True
            If True the data will be balanced.
        balance_limit: int, default=1000
            The limit of samples per label. If the number of samples is higher than the limit the data will be downsampled.
        """
        df_combined = self.con.execute("SELECT * FROM combinedTable").fetch_df()
        df_combined.drop(columns=["Combined"], inplace=True)

        # The following step is needed to get both Keyphrases and Classificaitons both as features in the Data object
        # Replace all ";;" values with "|" in the Classifications column
        df_combined["Classifications"] = df_combined["Classifications"].str.replace(";;", "|")
        
        # To reduce the number of rows for testing
        if sample is not None:
            df_combined = df_combined.sample(sample)
            
        if balanced:
            df_combined = balance_data(df_combined, balance_limit)
        
        self.df_combined = df_combined
            
        return df_combined
    
    def create_hetero_data(self, df_combined=None):
        if df_combined is not None:
            self.df_combined = df_combined
        
        self.data_model_preprocessor.load_data(self.df_combined)
        data = self.data_model_preprocessor.get_data()
        self.hetero_data = data
        torch.save(data, "Data/TextilePatents/raw/data.pt")
    
    def visualize_data(self, hetero_data=None):
        if hetero_data is not None:
            self.hetero_data = hetero_data
            
        if 'hetero_data' in self:
            HeteroDataVisualizer.visualize(self.hetero_data)
        else:
            print("HeteroData doesn't exist")
            
    def clean_processed_folder(self):
        """
        Clean the processed folder by removing all files and directories within it.
        """
        processed_folder = "Data\TextilePatents\processed"
        if os.path.exists(processed_folder):
            shutil.rmtree(processed_folder)
            os.makedirs(processed_folder)
            
    def clean_raw_folder(self):
        """
        Clean the raw folder by removing all files and directories within it.
        """
        raw_folder = "Data\TextilePatents\raw"
        if os.path.exists(raw_folder):
            shutil.rmtree(raw_folder)
            os.makedirs(raw_folder)
            
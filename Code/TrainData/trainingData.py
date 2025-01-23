import torch
from .visualize.visualizeHeteroData import HeteroDataVisualizer


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
    
    
    def create_balanced_data(self, sample=None):
        """
        Balanced data where all labels are equaly represented.
        
        args:
        sample: int, default=None
            Number of rows to sample from the data. Needed for testing or visualisation purposes.
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
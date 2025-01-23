import pandas as pd

def create_and_populate_table(file_name, table_name, con):
    input_file = "Data/CSVBaseData/"+ file_name+ ".csv"
    output_file = "Data/CSVBaseData/preprocessed/"+file_name+"-Text-Only.csv"

    df = pd.read_csv(input_file)

    columns_to_keep = ["Lens ID"]
    df["Combined"] = df["Title"] + " " + df["Abstract"].fillna("")
    df["Classifications"] = df["CPC Classifications"].fillna("") + ";;" + df["IPCR Classifications"].fillna("")
    df["Keyphrases"] = ""
    df["Category"] = table_name
    columns_to_keep.append("Combined")
    columns_to_keep.append("Keyphrases")
    columns_to_keep.append("Classifications")
    columns_to_keep.append("Category")
    df_filtered = df[columns_to_keep]

    # Save the filtered DataFrame to a new CSV file
    df_filtered.to_csv(output_file, index=False)

    con.execute("CREATE TABLE "+ table_name + " AS FROM read_csv('" + output_file + "')")


def clean_data_and_inser_into_db(file_names, table_names, con):
    """
    Takes the file names and table names and creates a table in the database with the given name and populates it with the data from the given file.
    Also creates a new column "Keyphrases" and "Category" in the table.
    
    Important: 
    - The file names and table names should be in the same order.
    - The table_nmes will be the labels for the classification later.
    
    Args:
    file_names: list of strings of data in the CSVBaseData folder. If a file is named "test.csv" the string should be "test"
    table_names: list of strings of the table names to be created in the database. Different because these will be the labels for the classification later.
    con: connection to the database
    """
    for i in range(len(file_names)):
        create_and_populate_table(file_names[i], table_names[i], con)
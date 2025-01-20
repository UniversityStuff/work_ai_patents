import pandas as pd
import duckdb as db

con = db.connect("Data/textile_patents.duckdb")

file_names = ["Anti-Seed-3k","Set-Mineral-Textile-Without-Metal", "Set-Metal-Textiles", "Set-Natural-Textiles", "Set-Synthetic-Textiles"]
table_names = ["antiSeed", "mineralTextiles", "metalTextiles", "naturalTextiles", "syntheticTextiles"]


def createAndPopulateTable(file_name, table_name):
    input_file = "Data/CSVBaseData/"+ file_name+ ".csv"
    output_file = "Data/CSVBaseData/"+file_name+"-Text-Only.csv"

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

for i in range(len(file_names)):
    createAndPopulateTable(file_names[i], table_names[i])

# antiSeed
# metalTextiles
# naturalTextiles
# mineralTextiles
# syntheticTextiles



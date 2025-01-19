import duckdb as db
from keyphraseExtractor import get_keyphrases

con = db.connect("Data/textile_patents.duckdb")

all_tables = ["antiSeed","metalTextiles","naturalTextiles", "mineralTextiles", "syntheticTextiles"]
# all_tables = ["metalTextiles"]


table_progress = 0
for table in all_tables:
    current_table = con.execute("SELECT * FROM " + table)
    progress = 0
    for row in current_table.fetch_df().iterrows():
        text = row[1]["Combined"]
        keyphrases = get_keyphrases(text)
        print(keyphrases)
        keyphrases = str(keyphrases)
        keyphrases.casefold()
        keyphrases = keyphrases.replace(',','')
        keyphrases = keyphrases.replace("""'""",'')
        # keyphrases = keyphrases.replace(" ", """', '""")
        keyphrases = keyphrases.replace("[", "'")
        keyphrases = keyphrases.replace("]", "'")
        keyphrases = keyphrases.replace(" ", "|")
        print(keyphrases)
        
        progress += 1
        print("Progress:")
        print(progress)
        con.execute("UPDATE " + table + " SET keyphrases = " + keyphrases + ''' WHERE "Lens ID" = \'''' + row[1]["Lens ID"] + "\'")
    print("Done Table:")
    print(table)
    table_progress += 1
    print("Table Progress:")
    print(table_progress/len(all_tables))
    
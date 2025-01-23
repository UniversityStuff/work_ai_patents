from .keyphraseExtractors.keyphraseExtractionKbirInspec import get_keyphrases
from .cleanData import clean_data_and_inser_into_db


def add_keyphrases_to_db(con, table_names):
    """
    Add keyphrases to the database.
    
    Exchange the keyphraseExtractor to use a different keyphrase extraction method.
    But be aware that you will have to adjust the following code.
    """
    table_progress = 0
    for table in table_names:
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
        print(table_progress/len(table_names))


def prepare_data_for_db(file_names, table_names, con):
    """
    Clean the data, insert it into the database and add keyphrases.
    It is possible to set the keyphraseExtractor here.
    
    Important:
    - The file names and table names should be in the same order.
    - The table_nmes will be the labels for the classification later.
    
    Args:
    file_names: list of strings
        List of file names to be cleaned and inserted into the database. If a file is named "test.csv" the string should be "test"
    table_names: list of strings
        List of table names in the database.
    con: duckdb.Connection
    """
    clean_data_and_inser_into_db(file_names, table_names, con)
    add_keyphrases_to_db(con, table_names)
    
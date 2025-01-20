import duckdb as db

con = db.connect("Data/textile_patents.duckdb")

con.execute("DELETE FROM testCombined")

# Insert some data
con.execute("INSERT INTO testCombined VALUES ('1', 'someText', 'Keyphrase1|Keyphrase2|Keyphrase3', 'Classificaton1;;Classificaton2;;Classificaton3', 'Category1')")
con.execute("INSERT INTO testCombined VALUES ('2', 'someText', 'Keyphrase1|Keyphrase2|Keyphrase3', 'Classificaton1;;Classificaton2;;Classificaton3', 'Category1')")
con.execute("INSERT INTO testCombined VALUES ('3', 'someText', 'Keyphrase4|Keyphrase5|Keyphrase6', 'Classificaton1;;Classificaton4;;Classificaton5', 'Category2')")
con.execute("INSERT INTO testCombined VALUES ('4', 'someText', 'Keyphrase4|Keyphrase5|Keyphrase6', 'Classificaton4;;Classificaton5', 'Category2')")
con.execute("INSERT INTO testCombined VALUES ('5', 'someText', 'Keyphrase7|Keyphrase8|Keyphrase9', 'Classificaton6;;Classificaton7;;Classificaton8', 'Category3')")
con.execute("INSERT INTO testCombined VALUES ('6', 'someText', 'Keyphrase7|Keyphrase8|Keyphrase9', 'Classificaton6;;Classificaton7;;Classificaton8', 'Category3')")
con.execute("INSERT INTO testCombined VALUES ('7', 'someText', 'Keyphrase3|Keyphrase11|Keyphrase12', 'Classificaton9;;Classificaton10;;Classificaton11', 'Category4')")
con.execute("INSERT INTO testCombined VALUES ('8', 'someText', 'Keyphrase9|Keyphrase11|Keyphrase12', 'Classificaton9;;Classificaton10;;Classificaton11', 'Category4')")

# Query the data
result = con.execute("SELECT * FROM testCombined")

# I want to get the columnnames of the result
print(result.df().head)
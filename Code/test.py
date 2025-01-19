import duckdb as db

con = db.connect("Data/textile_patents.duckdb")

result = con.execute("SELECT * FROM antiSeed")

# I want to get the columnnames of the result
print(result.df().head)
# Order of running the code

1. Clean Data
This removes the unwanted columns and combines the string columns to form a single column. The output is stored in the database.
2. prepareData
This extracts the keyphrases from the text and stores it in the database.
3. loadData
This loads the data from the database and stores it in a dataframe.
4. trainModel
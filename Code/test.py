from Code.PrepareData.prepareDataForDB import prepare_data_for_db
import duckdb as db



con = db.connect("Data/textile_patents.duckdb")


mineral_textile_test_set_file_names = ["mineral-textile"]
mineral_textile_test_set_table_names = ["mineralTextilesTestSet"]

# prepare_data_for_db(mineral_textile_test_set_file_names, mineral_textile_test_set_table_names, con)

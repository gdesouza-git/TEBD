# Get the database using the method we defined in pymongo_test_insert file
from pymongo_get_database import get_database
import random

dbname = get_database()
collection_name = dbname["tebd_col"]
id = random.randint(1, 1000000)
def insert(item):
  insert_json = {
    "_id" : id,
    "text" : item
  }

  collection_name.insert_many([insert_json])
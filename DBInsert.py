# Get the database using the method we defined in pymongo_test_insert file

from GetDatabase import get_database
from pymongo import MongoClient

dbname = get_database()
collection_name = dbname["tebd_col"]

def insert(item):
  collection_name.insert_many([item])
  print("- Registro inserido no banco")

def getEntry(id):
  CONNECTION_STRING = "mongodb://20.124.3.188:27017/"
  client = MongoClient(CONNECTION_STRING)
  db = client['tebd']
  col = db['tebd_col']

  x = list(col.find({"id": {"$eq": int(id)}}))
  x = str(x)
  return x
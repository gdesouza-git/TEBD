from pymongo import MongoClient
from dotenv import load_dotenv
import os

def get_database():
    load_dotenv()
    #os.environ['CONNECTION_STRING'] = os.getenv('CONNECTION_STRING')
    # Provide the mongodb string to connect python to mongodb using pymongo
    CONNECTION_STRING = "mongodb://20.124.3.188:27017/"
    # Create a connection using MongoClient
    client = MongoClient(CONNECTION_STRING)

    # Create the database for our example
    return client['tebd']


# This is added so that many files can reuse the function get_database()
if __name__ == "__main__":
    # Get the database
    dbname = get_database()
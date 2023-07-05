from pymongo import MongoClient

def get_database():
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
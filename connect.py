import pymongo
import json
from bson import json_util

class MongoDbConnector:
    def __init__(self):
        self.uri='REMOVED'
    def concatenate(self,json_files, output_file):
        combined_data = []

        for file_path in json_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    combined_data.extend(data)
                else:
                    combined_data.append(data)

        with open(output_file, 'w') as out_f:
            json.dump(combined_data, out_f, indent=2)

        print(f"Successfully concatenated {len(json_files)} files into '{output_file}'")

    def get(self,collection_name,db_name='OriginalData'):
        client = pymongo.MongoClient(self.uri)
        db = client[db_name]
        collection = db[collection_name]

        data = list(collection.find({}))
        return json.loads(json_util.dumps(data))

    def upload(self,json_file_path, collection_name, db_name='OriginalData'):
        client = pymongo.MongoClient(self.uri)
        db = client[db_name]
        collection = db[collection_name]

        with open(json_file_path, 'r') as datafile:
            data = json.load(datafile)

        if isinstance(data, list):
            collection.insert_many(data)
        else:
            collection.insert_one(data)

        print(f"Successfully uploaded data to collection: {collection_name}")

def main():
    obj = MongoDbConnector()
    file = obj.get('Software')
    print(file)




if __name__ == '__main__':
    main()
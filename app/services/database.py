from datetime import datetime

from pymongo import MongoClient

from app.config import settings
from app.services.embedding import model

class Database:
    def __init__(self):
        print('starting database')
        self._client = MongoClient(settings.mongodb_url, serverSelectionTimeoutMS=5000)
        self._db = self._client[settings.database_name]
        self._collections = self._db[settings.collections]
        self.counters = self._db.counters

    def add_new_collection(self, collection_name):
        self._collections.insert_one({
            'name': collection_name,
            'date': datetime.now()
        })
        self.counters.insert_one({
            '_id': collection_name,
            'seq': 0
        })
        face_data = model.process_image(settings.image)
        person_id = self._increment_counter(collection_name)
        self.add_new_face_to_collection(face_data, settings.image, person_id, collection_name)


    def _increment_counter(self, counter_id):
        return self.counters.find_one_and_update(
            {'_id': counter_id},
            {'$inc': {'seq': 1}},
            upsert=True,
            return_document=True
        )['seq']

    def get_increment_counter(self, counter_id):
        return self.counters.find_one({'_id': counter_id})['seq']

    def add_new_face_to_collection(self, face_data, image_path, person_id, collection_name):
        if self._collections.find({'name': collection_name}):
            face_id = self._increment_counter(collection_name)
            client_data = {
                '_id': face_id,
                'person_id': person_id,
                'embedding': face_data.embedding.tolist(),
                'pose': face_data.pose.tolist(),
                "gender": int(face_data.gender),
                "age": int(face_data.age),
                'image_path': image_path,
                'date': datetime.now()
            }
            self._db[collection_name].insert_one(client_data)
            return face_id
        else:
            return False

    def get_docs_from_collection(self, collection_name):
        if self._collections.find({'name': collection_name}):
            return self._db[collection_name].find()
        else:
            return False

    def get_collections_names(self):
        return [doc['name'] for doc in self._collections.find()]

    def delete_collection(self, collection_name):
        if self._collections.find({'name': collection_name}):
            self._collections.delete_one({'name': collection_name})
            self.counters.delete_one({'_id': collection_name})
            self._db[collection_name].delete_many({})

    def delete_face(self, collection_name, face_id):
        if self._collections.find({'name': collection_name}):
            self._db[collection_name].delete_one({'_id': face_id})




db = Database()
import faiss
import numpy as np

from app.services.database import db


class FaceRecognition:
    def __init__(self):
        print("Starting FaceRecognition")
        self.dimension = 512
        self._indexes = {
            db_name: faiss.IndexFlatIP(self.dimension)
            for db_name in db.get_collections_names()
        }
        self._indices = {db_name: list() for db_name in db.get_collections_names()}
        self.load_indices()

    def load_indices(self):
        for collection in self._indexes.keys():
            existing_faces = db.get_docs_from_collection(collection)
            # print(existing_faces)
            for doc in existing_faces:
                embedding = np.array(doc["embedding"], dtype=np.float32)
                self._indices[collection].append(doc["person_id"])
                vectors = np.array([embedding]).astype("float32")
                if vectors.ndim != 2:
                    raise ValueError("Vectors should be a 2D array")
                faiss.normalize_L2(vectors)
                self._indexes[collection].add(vectors)

    def search_face(
        self,
        face_data,
        db_name,
    ):
        query = np.array(face_data.embedding).astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)

        scores, ids = self._indexes[db_name].search(query, 1)
        if len(scores) == 0 or len(ids) == 0 or len(ids[0]) == 0:
            return 0, 0
        person_id = int(self._indices[db_name][ids[0][0]])

        return abs(round(scores[0][0] * 100, 3)), person_id

    def add_to_index(self, embedding, db_name, face_id):
        vectors = np.array([embedding]).astype("float32")
        faiss.normalize_L2(vectors)
        self._indexes[db_name].add(vectors)
        self._indices[db_name].extend([face_id])

    def create_index(self, collection_name):
        if collection_name is db.get_collections_names():
            raise ValueError(f"Index for collection {collection_name} already exists")
        else:
            db.add_new_collection(collection_name)
        existing_faces = db.get_docs_from_collection(collection_name)
        embeddings = []
        for doc in existing_faces:
            embeddings.append(np.array(doc["embedding"], dtype=np.float32))
            self._indices[collection_name] = [doc["person_id"]]
        vectors = np.array(embeddings).astype("float32")
        if vectors.ndim != 2:
            raise ValueError("Vectors should be a 2D array")
        faiss.normalize_L2(vectors)
        self._indexes[collection_name] = faiss.IndexFlatIP(self.dimension)
        self._indexes[collection_name].add(vectors)

    def delete_index(self, collection_name):
        if collection_name not in self._indexes.keys():
            raise ValueError(f"Index for collection {collection_name} does not exist")
        else:
            del self._indexes[collection_name]
            del self._indices[collection_name]

    def update_index(self):
        self._indexes.clear()
        self._indices.clear()
        self._indexes = {
            db_name: faiss.IndexFlatIP(self.dimension)
            for db_name in db.get_collections_names()
        }
        self._indices = {db_name: list() for db_name in db.get_collections_names()}
        self.load_indices()


face_recognition = FaceRecognition()

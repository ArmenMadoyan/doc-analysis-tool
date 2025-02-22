from pathlib import Path
from typing import Any
import chromadb
from anyio.streams import file
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
import numpy as np
import json

class Parser:

    def __init__(self, save_path:str = 'parsed_text.json'):

        self.file_path = None
        self.chunk_size = None
        self.pdf_text = None
        self.chunked_text = {}
        self.parsed_text_ = {}
        self.embeddings = {}
        self.collection = None
        self.save_path = save_path

    def parse_folder(self, file_path:str)-> dict:
        self.file_path = file_path
        for file_name in Path(file_path).glob("*.pdf"):
            self.parsed_text_[str(file_name)] = extract_text(file_name)
            print(f'{file_name} parsed')

        with open(self.save_path, "w", encoding="utf-8") as json_file:
            json.dump(self.parsed_text_, json_file, ensure_ascii=False, indent=4)

        return self.parsed_text_

    def retrieve_text(self):
        with open(self.save_path, "r", encoding="utf-8") as f:
            self.pdf_text = json.load(f)
        return self.pdf_text

    def chunk_text(self, file_text:dict, chunk_size = 500) -> dict:
        self.chunk_size = chunk_size
        for filename, text in file_text.items():
            print(f'{filename} chunked')
            words = text.split()
            self.chunked_text[filename] = [" ".join(words[i:i + self.chunk_size]) for i in range(0, len(words), self.chunk_size)]

        return self.chunked_text

    def create_embeddings(self, chunked_text:dict, model: SentenceTransformer) -> dict:
        self.embeddings = {filename: np.array(model.encode(text)) for filename, text in chunked_text.items() }
        return self.embeddings

    def store_embeddings(self, embeddings:dict, path:str, collection_name:str) -> Any:
        chroma_db = chromadb.PersistentClient(path=path)
        chroma_db.delete_collection(collection_name)
        self.collection = chroma_db.get_or_create_collection(collection_name)

        for i, (filename, embedding_array) in enumerate(embeddings.items()):
            embedding_array = np.array(embedding_array)  # Ensure it's a NumPy array
            if embedding_array.ndim == 1:  # Convert single embeddings into a proper 2D shape
                embedding_array = embedding_array.reshape(1, -1)

            # Generate unique IDs for each chunked embedding
            ids = [f"{i}_{j}" for j in range(len(embedding_array))]

            # Convert embeddings to list format
            embeddings_list = embedding_array.tolist()

            # Create metadata for each embedding chunk
            metadatas = [{"filename": filename, "chunk": j} for j in range(len(embedding_array))]

            # Store embeddings correctly
            self.collection.add(
                ids=ids,  # List of unique IDs
                embeddings=embeddings_list,  # List of embeddings
                metadatas=metadatas  # List of metadata entries
            )


if __name__ == "__main__":
    ps = Parser()
    file_path = "./PoliciesForTheTask"
    # ps.parse_folder(file_path)
    parsed_dict = ps.retrieve_text()
    # print(parsed_dict)
    chunked = ps.chunk_text(parsed_dict)
    # print(chunked)
    embeddings = ps.create_embeddings(chunked_text=chunked,model=SentenceTransformer("all-MiniLM-L6-v2"))
    # for filename, embedding in ps.embeddings.items():
    #     print(ps.embeddings[filename])

    path = "./chroma"
    collection_name = "policy_embeddings"
    ps.store_embeddings(embeddings, path, collection_name)

from pathlib import Path
from typing import Any
import chromadb
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import chunker
import os

class Parser:

    def __init__(self):

        self.folder_path = None
        self.chunk_size = None
        self.chunked_text = {}
        self.parsed_text_ = {}
        self.embeddings = {}
        self.collection = None
        self.parser_model = SentenceTransformer("all-MiniLM-L6-v2")

    def parse_folder(self, folder_path:str, chroma_path:str, collection_name:str, parser_model)-> dict:
        self.folder_path = folder_path

        for file_name in Path(self.folder_path).glob("*.pdf"):
            self.parsed_text_[str(file_name)] = extract_text(file_name)
            print(f'{file_name} parsed')

        chunk_processor = chunker.TextProcessor(chunk_size=500, max_words=30000)
        self.chunked_text = chunk_processor.chunk_text(self.parsed_text_)
        # self.chunked_text = self.chunk_text(self.parsed_text_)
        self.embeddings = self.create_embeddings(self.chunked_text, model= parser_model)
        print('Files are Embedded!!!')
        self.store_embeddings(self.embeddings, self.chunked_text, path=chroma_path, collection_name=collection_name)
        print('Embeddings are Stored!!!')
        return self.parsed_text_

    def chunk_text(self, file_text:dict, chunk_size = 500) -> dict:
        self.chunk_size = chunk_size
        chunked_text = {}
        for filename, text in file_text.items():
            print(f'{filename} chunked')
            words = text.split()
            chunked_text[filename] = [" ".join(words[i:i + self.chunk_size]) for i in range(0, len(words), self.chunk_size)]

        return chunked_text

    @staticmethod
    def create_embeddings(chunked_text:dict, model: SentenceTransformer) -> dict:
        return {filename: np.array(model.encode(text)) for filename, text in chunked_text.items() }

    def store_embeddings(self, embeddings:dict, text_chunks:dict, path:str, collection_name:str) -> Any:
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

            chunk_texts = text_chunks.get(filename, [])

            if len(chunk_texts) != len(embedding_array):
                print(
                    f"Warning: Mismatch in embeddings ({len(embedding_array)}) and text chunks ({len(chunk_texts)}) for {filename}")
            # Create metadata for each embedding chunk
            metadatas = [{"filename": filename, "chunk": j} for j in range(len(embedding_array))]

            # Store embeddings correctly
            self.collection.add(
                ids=ids,  # List of unique IDs
                embeddings=embeddings_list,  # List of embeddings
                metadatas=metadatas,   # List of metadata entries
                documents = chunk_texts
            )


if __name__ == "__main__":
    ps = Parser()
    folder_path = str(os.getenv('FILE_PATH'))
    ps.parse_folder('./PoliciesForTheTask')

    # parsed_dict = ps.retrieve_text()
    # print(parsed_dict)
    # chunked = ps.chunk_text(parsed_dict)
    # print(chunked)
    # embeddings = ps.create_embeddings(chunked_text=chunked,model=SentenceTransformer("all-MiniLM-L6-v2"))
    # for filename, embedding in ps.embeddings.items():
    #     print(ps.embeddings[filename])
    #
    # path = "./chroma"
    # collection_name = "policy_embeddings"
    # ps.store_embeddings(embeddings, chunked, path, collection_name)

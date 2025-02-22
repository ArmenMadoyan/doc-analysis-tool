# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

file_path = "./PoliciesForTheTask"
model = SentenceTransformer("all-MiniLM-L6-v2")

path = "./chroma"
collection_name = "policy_embeddings"
chroma_db = chromadb.PersistentClient(path=path)
# chroma_db.delete_collection("policy_embeddings")
chroma_collection = chroma_db.get_or_create_collection(collection_name)
# query = "risks and recommendations."
# query_embedding = model.encode(query).tolist()

# Retrieve similar documents
# results = chroma_collection.query(
#     query_embeddings=[query_embedding],
#     n_results=3,
#     include=["embeddings", "documents", "metadatas", "distances"],# Get top 3 matches
# )
#
# print("Top matches:", results["embeddings"])

# print(chroma_collection.get(include=['embeddings', 'documents', 'metadatas']))

# texts = ["Anti-bribery policy", "Whistleblowing procedures", "Code of conduct"]
# embeddings = model.encode(texts)

# Compute similarity
# query_embedding = model.encode("bribery guidelines")
# similarities = np.dot(embeddings, query_embedding)


# print("Most relevant:", texts[np.argmax(similarities)])

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define policy categories & associated key data points
categories = {
    "Anti-Corruption and Anti-Bribery": [
        "anti-corruption", "bribery", "fraud prevention", "whistleblowing",
        "due diligence", "sanctions", "legal cases"
    ],
    "Ethical Business Practices": [
        "code of conduct", "compliance system", "ethical decision-making",
        "non-compliance incidents"
    ],
    "Lobbying and Political Contributions": [
        "lobbying activities", "political contributions",
        "transparency", "governance oversight"
    ],
    "Additional Governance Disclosures": [
        "alignment with standards", "stakeholder engagement",
        "performance metrics", "assurance"
    ]
}

# Query ChromaDB for each category
retrieved_data = {}

for category, keywords in categories.items():
    category_texts = []
    for keyword in keywords:
        query_embedding = model.encode(keyword).tolist()
        results = chroma_collection.query(query_embeddings=[query_embedding], n_results=3)  # Get top 3 matches

        for match in results["documents"][0]:  # Extract matched documents
            if match not in category_texts:
                category_texts.append(match)

    if category_texts:
        retrieved_data[category] = " ".join(category_texts)  # Combine retrieved text

# Notify about missing categories
missing_categories = [category for category in categories if category not in retrieved_data]
if missing_categories:
    print(f"Missing categories: {missing_categories}")

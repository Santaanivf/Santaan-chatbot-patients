from langchain.embeddings import HuggingFaceEmbeddings


print("Initializing HuggingFace embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Model initialized successfully.")

# query = "Hi How are you?"
# print(f"Embedding query: '{query}'")
# embed_vec = embeddings.embed_query(query)
# print("Embeddings generated successfully.")

# print(f"Embedding vector: {embed_vec}")
# print(f"Length of the embedding vector: {len(embed_vec)}")

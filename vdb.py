from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from chunker import docs
from embedder import embeddings
from dotenv import load_dotenv
import os

# Load environment variables from .env file
# load_dotenv()
# pinecone_api_key = os.getenv("PINECONE_API_KEY")
PINECONE_API_KEY="pcsk_4FN6jn_7qPJVojUJwdx83vKckuzRxNjUfR3hb5Vqa1gWwJMcCoJyaj5MstK2byak7at1yT"
# Debug: Check if API key is loaded
# if not pinecone_api_key:
#     raise ValueError("PINECONE_API_KEY is not loaded from .env file. Check your setup.")

# Define index name
index_name = "santaan-material"

# Initialize Pinecone client
print("Initializing Pinecone client...")
pc = Pinecone(api_key=PINECONE_API_KEY)  # Correct variable used
print("Pinecone client initialized.")


# Function to create an index and conditionally upsert embeddings
def create_index_and_upsert(index_name):
    print(f"Checking if index '{index_name}' exists...")
    # Check if the index exists
    if index_name not in pc.list_indexes().names():
        print(f"Index '{index_name}' does not exist. Creating it now...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",  
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print(f"Index '{index_name}' created successfully.")
        
        # Create Pinecone vector store and upsert embeddings
        print(f"Creating vector store and upserting embeddings for index '{index_name}'...")
        docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
        print("Vector store created and embeddings upserted successfully.")
    else:
        print(f"Index '{index_name}' already exists. Skipping upsertion of embeddings.")

# Call the function to handle index creation and conditional upsertion
create_index_and_upsert(index_name)

# Debugging: Print loaded documents
print(f"Number of documents loaded: {len(docs)}")
for i, doc in enumerate(docs[:5]):  # Print the first 5 documents
    print(f"Document {i+1}: {doc.page_content[:100]}...")  # Print the first 100 characters

# Debugging: Print embedding model information
print(f"Embedding model used: {embeddings.model_name}")

# Perform similarity search
query = "What is IVF?"
print(f"Performing similarity search for query: '{query}'")
# Initialize vector store for search if the index exists
if index_name in pc.list_indexes().names():
    docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    r_docs = docsearch.similarity_search(query=query, k=2)
    print(f"Number of similar documents found: {len(r_docs)}")

    # Print retrieved documents
    for i, doc in enumerate(r_docs):
        print(f"Retrieved Document {i+1}: {doc.page_content}...")
else:
    print("Index not found. Cannot perform similarity search.")

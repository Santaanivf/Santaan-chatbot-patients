from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFChunker:
    def __init__(self, pdf_path, chunk_size=120, chunk_overlap=30) -> None:
        print(f"Initializing PDFChunker with file: {pdf_path}")
        # Load and split the PDF into pages
        self.pages = PyPDFLoader(pdf_path).load_and_split()
        print(f"Number of pages loaded: {len(self.pages)}")
        
        # Set up chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"Chunk size: {chunk_size}, Chunk overlap: {chunk_overlap}")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def chunk(self) -> List:
        print("Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(self.pages)
        print(f"Number of chunks created: {len(chunks)}")
        return chunks


# Use the PDFChunker
chunker = PDFChunker("Resources/v1_v2_merged.pdf")

# Get the chunks
docs = chunker.chunk()

# Print the chunks for better understanding (optional: can be removed in production)
print(f"First chunk: {docs[0] if docs else 'No chunks created'}")

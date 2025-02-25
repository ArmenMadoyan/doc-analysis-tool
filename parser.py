import os
import pytesseract
import pdfplumber
import tempfile
import ocrmypdf
import shutil
from PIL import Image
import docx2txt
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import config


class Parser:

    def __init__(self):
        self.directory = None

    def parse_pdf(self, file_path):
        """Extract text from a PDF (either searchable or scanned)."""
        text = ""

        try:
            # Try extracting text from searchable PDFs
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""  # Ensure it doesn't break on None

            if text.strip():  # If we got text, return it
                return text

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                temp_pdf_path = temp_pdf.name

            try:
                # Run OCR on the PDF and save the output to temp file
                ocrmypdf.ocr(file_path, temp_pdf_path, deskew=True, force_ocr=True, language="eng")

                # Read extracted text from the processed PDF
                with open(temp_pdf_path, "rb") as f:
                    from PyPDF2 import PdfReader
                    reader = PdfReader(f)
                    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

                return text

            finally:
                # Clean up the temporary file
                shutil.rmtree(temp_pdf_path, ignore_errors=True)

        except Exception as e:
            print(f"Error parsing PDF {file_path}: {e}")
            return None


    def parse_image(self, file_path):
        """Extract text from an image using OCR."""
        try:
            image = Image.open(file_path)
            return pytesseract.image_to_string(image)
        except Exception as e:
            print(f"Error parsing image {file_path}: {e}")
            return None

    def parse_docx(self, file_path):
        """Extract text from a Word document."""
        try:
            doc = docx2txt.process(file_path)
            return doc
        except Exception as e:
            print(f"Error parsing Word document {file_path}: {e}")
            return None

    def parse_file(self,file_path):
        """Determine file type and parse accordingly."""
        ext = file_path.lower().split('.')[-1]

        if ext == "pdf":
            return self.parse_pdf(file_path)
        elif ext in ["png", "jpg", "jpeg", "tiff"]:
            return self.parse_image(file_path)
        elif ext in ["docx"]:
            return self.parse_docx(file_path)
        else:
            print(f"Unsupported file format: {file_path}")
            return None

    def process_directory(self, directory):
        """Parse all supported files in a directory and return LangChain-compatible Documents."""
        langchain_docs = []

        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                text = self.parse_file(file_path)
                if text:
                    text = self.clean_text(text)
                    langchain_docs.append(Document(page_content=text, metadata={"source": file_path}))
                # print(f"Processed {text}")

        return langchain_docs

    def clean_text(self, text: str) -> str:

        # Remove extra newlines, headers, bullet points etc.
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"[-●•◦○]", "", text)
        text = re.sub(r"Page \d+|Table of Contents|\.{3,}", "", text)
        return text.strip()

    def store_in_chroma(self, directory, embeddings):
        self.docs = self.process_directory(directory)
        print(f"Loaded {len(self.docs)} documents into LangChain.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # approx. 1000 characters per chunk (adjust as needed)
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(self.docs)
        print(f"Split into {len(split_docs)} chunks.")

        persist_directory = "chroma_db"  # directory for persistent storage

        # Create (or load) the Chroma vector store.
        vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_directory)
        print("Documents stored in Chroma.")

        return vectorstore

def main():

    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_DIR
    FOLDER_PATH = config.FILE_DIR

    ps = Parser()
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    ps.store_in_chroma(FOLDER_PATH, embeddings)


if __name__ == "__main__":
    main()




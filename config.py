import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env if available

FILE_DIR = os.getenv("FILE_DIR")
TESSERACT_DIR = os.getenv("TESSERACT_DIR")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

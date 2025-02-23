import nltk
import re
from nltk.tokenize import sent_tokenize
from typing import Dict, List

nltk.download('punkt')  # Ensure sentence tokenizer is available

class TextProcessor:
    def __init__(self, chunk_size=500, max_words=30000):
        self.chunk_size = chunk_size
        self.max_words = max_words
        self.chunked_text = {}

    def clean_text(self, text: str) -> str:
        """Removes extra whitespace, bullet points, and meaningless characters."""
        text = re.sub(r"\n+", " ", text)  # Remove newlines
        text = re.sub(r"\s{2,}", " ", text)  # Remove extra spaces
        text = re.sub(r"[-●•◦]", "", text)  # Remove bullet points
        text = re.sub(r"Page \d+|Table of Contents|\.{3,}", "", text)  # Remove headers
        return text.strip()

    def chunk_text(self, file_text: Dict[str, str]) -> Dict[str, List[str]]:
        """Chunks text into meaningful sentences while respecting API limits."""
        for filename, text in file_text.items():
            print(f"Processing {filename}...")

            cleaned_text = self.clean_text(text)
            sentences = sent_tokenize(cleaned_text)  # Tokenize into sentences

            chunks = []
            current_chunk = []
            word_count = 0
            total_word_count = 0  # To ensure we don't exceed API limits

            for sentence in sentences:
                words = sentence.split()
                sentence_length = len(words)

                if total_word_count + sentence_length > self.max_words:
                    break  # Stop if we exceed API limits

                if word_count + sentence_length > self.chunk_size:
                    chunks.append(" ".join(current_chunk))  # Save current chunk
                    current_chunk = []
                    word_count = 0

                current_chunk.append(sentence)
                word_count += sentence_length
                total_word_count += sentence_length

            if current_chunk:  # Save any remaining chunk
                chunks.append(" ".join(current_chunk))

            self.chunked_text[filename] = chunks

        return self.chunked_text

if __name__ == "__main__":
    # Example usage
    file_text = {
        "Anti-Corruption Policy.txt": "Your long text here...",
        "Whistleblowing Policy.txt": "Another long text..."
    }

    processor = TextProcessor(chunk_size=500, max_words=30000)
    chunked_data = processor.chunk_text(file_text)

    # Output the chunked data
    for filename, chunks in chunked_data.items():
        print(f"\n{filename} has {chunks} chunks")

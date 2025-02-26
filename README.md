# Doc Analysis Tool Project

This project implements a Retrieval-Augmented Generation (RAG) pipeline using LangChain, ChromaDB, and multiple LLM APIs (OpenAI, Gemini) to process, extract, and summarize compliance-related documents. The system loads documents from a specified folder, processes them with OCR (using OCRmyPDF for scanned PDFs), stores embeddings in ChromaDB, and generates structured reports based on predefined categories.

## Project Structure

```
my_project/
├── .env                   # Auto-generated configuration file (created on first run)
├── config.py              # Loads environment variables for use across the project
├── Parser.py              # Parses and extracts text from documents in the target folder
├── chained.py             # Additional processing logic (e.g., OCR, chaining operations)
├── main.py                # Main orchestrator: prompts for configuration, runs Parser.py, chained.py, then main processing
├── requirements.txt       # List of required packages
└── README.md              # This file
```

## Requirements

Install all necessary dependencies using:

```bash
pip install -r requirements.txt
```

> **Note:** You may need additional system dependencies for `ocrmypdf` and Tesseract OCR.

## Running the Project

To run the project, execute:

```bash
python main.py
```

When you run `main.py`, the following steps occur in sequence:

1. **Configuration Prompt:**  
   If no `.env` file is present, you'll be prompted for the folder path, Tesseract path, and API keys. These values are stored in `.env` and loaded by all modules via `config.py`.

2. **Parser Execution:**  
   The `Parser.py` script is executed to parse documents from the provided folder.

3. **Chained Processing:**  
   The `chained.py` script runs next, performing additional processing (e.g., OCR using OCRmyPDF).

4. **Main Processing:**  
   Finally, `main.py` continues with its own processing logic (e.g., storing embeddings, retrieving relevant document chunks, generating structured reports).

## Generating Reports and Evaluation

The project generates structured reports based on the following predefined categories and data points:

- **Anti-Corruption and Anti-Bribery:**  
  Policies, training, whistleblowing mechanisms, due diligence, legal cases, sanctions.

- **Ethical Business Practices:**  
  Code of conduct, compliance system, ethical decision-making, non-compliance incidents.

- **Lobbying and Political Contributions:**  
  Lobbying activities, political contributions, transparency, governance oversight.

- **Additional Governance Disclosures:**  
  Alignment with standards, stakeholder engagement, performance metrics, assurance.

The system retrieves relevant document chunks from ChromaDB using semantic similarity search and then uses multiple LLM APIs e.g GPT-4-turbo, Gemini (In future might be also Claude, Llama, Mistral) to generate and evaluate the final report. Gemini is used as a judge LLM to assess report quality, accuracy, and completeness.

## Optimization Ideas

- **Chunking Parameters:**  
  Experiment with different chunk sizes (e.g., 500–1500 tokens) and overlap values to maintain context without redundant information.

- **Embedding Models:**  
  Compare results using different embedding models (e.g., OpenAI’s text-embedding-ada-002 versus local models like BERT).

- **Retrieval Tuning:**  
  Fine-tune the similarity threshold and number of retrieved chunks to optimize relevance.

- **Prompt Engineering:**  
  Adjust and refine prompts for both report generation and evaluation to ensure clarity and focus.

- **Model Ensemble:**  
  Consider combining outputs from multiple LLMs or using a multi-step chain-of-thought evaluation for more robust results.

- **Monitoring:**  
  Use LangSmith or other monitoring tools to track token usage, latency, and performance metrics for further optimizations.

## Contributing

Contributions and suggestions are welcome! Feel free to open issues or submit pull requests.

## License

[MIT License](LICENSE)

---

This README provides a complete overview of your RAG project setup, installation, configuration, and running instructions, along with optimization ideas. Happy coding!

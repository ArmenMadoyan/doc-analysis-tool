import os
from typing import Any, Dict, List, Optional, Mapping

# STEP 1: Load PDFs from a directory
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

# Load all PDFs from the given directory (change the path as needed)
pdf_dir = "./PoliciesForTheTask"  # <-- update to your directory
loader = DirectoryLoader(pdf_dir, glob="*.pdf", loader_cls = PyPDFLoader)
documents = loader.load()
print(f"Loaded {len(documents)} documents.")

# STEP 2: Split documents into chunks with cleaning
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

def clean_text(text: str) -> str:
    # Remove extra newlines, headers, bullet points etc.
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[-●•◦○]", "", text)
    text = re.sub(r"Page \d+|Table of Contents|\.{3,}", "", text)
    return text.strip()


# First, clean each document’s page content (if needed)
for doc in documents:
    doc.page_content = clean_text(doc.page_content)

# Now split documents; adjust chunk_size and overlap as needed.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # approx. 1000 characters per chunk (adjust as needed)
    chunk_overlap=200
)
split_docs = text_splitter.split_documents(documents)
print(f"Split into {len(split_docs)} chunks.")


# STEP 3: Store chunks in ChromaDB (vector store)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Use a cost-efficient embedding model; you can also use SentenceTransformersEmbeddings, etc.
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
persist_directory = "chroma_db"  # directory for persistent storage

# Create (or load) the Chroma vector store.
vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_directory)
print("Documents stored in Chroma.")

# STEP 4: Retrieve similar documents/chunks given a query
query = "summarize the anti-corruption measures and whistleblowing procedures"
retrieved_docs = vectorstore.similarity_search_with_score(query, k=3)
print("Retrieved the following document chunks:")

for doc, score in retrieved_docs:
    print(f"Score: {score:.4f} | Metadata: {doc.metadata}")
    print(f"Content excerpt: {doc.page_content[:200]}...\n")

CATEGORIES = {
    "Anti-Corruption and Anti-Bribery": [
        "policies", "training", "whistleblowing mechanisms",
        "due diligence", "legal cases", "sanctions"
    ],
    "Ethical Business Practices": [
        "code of conduct", "compliance system",
        "ethical decision-making", "non-compliance incidents"
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

# STEP 5: Generate a summary/answer using an LLM chain
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
import textwrap
from langchain.prompts import PromptTemplate


def generate_structured_report(llm, categories_dict: Dict[str, List[str]]) -> str:
    """
    For each category -> data point, we retrieve relevant text from Chroma,
    then ask the LLM to summarize. If no results, we mark it as missing.
    """
    final_report = []
    missing_points = []

    for category, data_points in categories_dict.items():
        category_content = []
        for dp in data_points:
            query = dp  # simplistic approach: direct search using data point
            results = vectorstore.similarity_search_with_score(query, k=3)

            # Filter out results with high distance or no relevant text
            # (Tune threshold based on your experiments)
            relevant_chunks = [doc for doc, score in results if score < 0.8]

            if relevant_chunks:
                # Combine chunk text
                combined_text = "\n".join([chunk.page_content for chunk in relevant_chunks])

                # Summarize chunk text using the LLM
                prompt_text = f""" You are an expert in compliance and policy summarization. Using the following extracted document chunks,
                Summarize the following text focusing specifically on '{dp}'.
                Text:
                {combined_text}
                """
                summary = llm.invoke(prompt_text)
                # print(summary)
                category_content.append(f"**{dp.title()}**:\n{summary}\n")
            else:
                missing_points.append(f"{category} - {dp}")

        # If we found content for this category, wrap it up
        if category_content:
            section_text = f"# {category}\n\n" + "\n".join(category_content)
            final_report.append(section_text)

    # Add missing info notice
    if missing_points:
        missing_str = "\n".join(missing_points)
        final_report.append(
            "## Missing Categories/Data Points\n"
            "The following data points were not found in the analyzed documents:\n"
            f"{missing_str}"
        )

    return "\n".join(final_report)


# EXAMPLE: Use an OpenAI LLM or any other LLM
my_llm = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0.3,
    api_key= os.getenv("OPENAI_API_KEY")
)

report = generate_structured_report(my_llm, CATEGORIES)
print(textwrap.shorten(report, width=2000, placeholder="..."))

# from langchain_anthropic import ChatAnthropic
#
# claude_llm = ChatAnthropic(
#     model="claude-3-sonnet-20240229",  # or 'claude-instant-1', etc.
#     api_key="YOUR_ANTHROPIC_KEY"
# )

# claude_report = generate_structured_report(claude_llm, CATEGORIES)
# print("Claude-based Report:\n", claude_report)


from langchain_huggingface import HuggingFacePipeline
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#
# model_id = "meta-llama/Llama-3.2-1B-Instruct"  # Example
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=1024, temperature=0.0)

# llama_llm = HuggingFacePipeline(pipeline=pipe)
#
# llama_report = generate_structured_report(llama_llm, CATEGORIES)
# print("LLama-based Report:\n", llama_report)


# STEP 6: Use LangSmith to monitor chain execution
# LangSmith monitoring can be enabled by setting the environment variable and using callbacks.
# from langchain_community.callbacks import get_openai_callback
#
# with get_openai_callback() as cb:
#     result = my_llm.invoke(documents=report)
#     print("Chain execution details:")
#     print(cb)


import google.generativeai as genai
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiJudge(LLM):
    @property
    def _llm_type(self) -> str:
        return "gemini-judge"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = genai.chat(
            model="gemini-pro",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.last  # Return final string

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": "gemini-pro"}

gemini_judge = GeminiJudge()

def judge_report(judge_llm: LLM, final_report: str) -> str:
    """
    Evaluate the final report based on attached example or desired criteria.
    """
    judge_prompt = f"""
    You are a compliance expert. Evaluate the following report for correctness,
    completeness, and alignment with the 4 categories:
    {final_report}

    Provide a score from 1-10 and a brief explanation.
    """
    return judge_llm(judge_prompt)

evaluation = judge_report(gemini_judge, report)
print("Gemini Judge Evaluation:\n", evaluation)


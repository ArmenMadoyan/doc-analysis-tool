import os
from typing import  Dict, List
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import json
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain_google_genai import GoogleGenerativeAI
import textwrap
import config


class GeminiJudge(LLM):
    @property
    def _llm_type(self) -> str:
        return "gemini-judge"

    def _call(self, prompt: str, stop=None) -> str:
        gemini_llm = GoogleGenerativeAI(
            model='gemini-1.5-pro',
            api_key=os.getenv('GEMINI_API_KEY'),
            temperature=0.3
        )
        messages = [("system", f"{prompt}")]
        response = gemini_llm.invoke(messages)

        return response  # Return the generated text

    @property
    def _identifying_params(self):
        return {"model": "gemini-pro"}

def main():
    persist_directory = "chroma_db"

    # Load all PDFs from the given directory (change the path as needed)
    vectorstore_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002")  # Add this to avoid errors
    )

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

    # Generate a summary/answer using an LLM chain

    def generate_structured_report(llm, vectorstore, categories_dict: Dict[str, List[str]]) -> str:
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
                relevant_chunks = [doc for doc, score in results if score < 0.5]

                if relevant_chunks:
                    # Combine chunk text
                    combined_text = "\n".join([chunk.page_content for chunk in relevant_chunks])

                    # Summarize chunk text using the LLM
                    prompt_text = f""" You are an expert in compliance and policy summarization. Using the following extracted document chunks.
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

    gpt_llm = ChatOpenAI(
        model="gpt-4-turbo",
        temperature=0.3,
        api_key= config.OPENAI_API_KEY
    )

    report_gpt = generate_structured_report(gpt_llm,vectorstore_db, CATEGORIES)
    print(textwrap.shorten(report_gpt, width=2000, placeholder="..."))

    gemini_llm = GoogleGenerativeAI(
        model='gemini-1.5-pro',
        api_key=config.GEMINI_API_KEY,
        temperature=0.3
    )
    report_gemini = generate_structured_report(gemini_llm, vectorstore_db, CATEGORIES)
    print(textwrap.shorten(report_gemini, width=2000, placeholder="..."))

    reports = {
        "gemini": report_gemini,
        "gpt": report_gpt
    }

    # Store the reports in a JSON file
    with open("reports.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=4)
    print("Reports stored in reports.json")


if __name__ == "__main__":
    main()

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import json
import os
import textwrap
from langchain.llms.base import LLM
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()


class GeminiJudge(LLM):
    @property
    def _llm_type(self) -> str:
        return "gemini-judge"

    def _call(self, prompt: str, stop=None) -> str:
        gemini_llm = GoogleGenerativeAI(
            model='gemini-pro',
            api_key=os.getenv('GEMINI_API_KEY'),
            temperature=0.3
        )
        messages = [("system", f"{prompt}")]
        response = gemini_llm.invoke(messages)

        return response  # Return the generated text

    @property
    def _identifying_params(self):
        return {"model": "gemini-pro"}


# Instantiate the Gemini judge
gemini_judge = GeminiJudge()


def evaluate_report(report: str, judge_llm: LLM) -> float:
    """
    Uses the judge LLM to score the report on accuracy, completeness, and clarity.
    Returns a score between 1 and 10.
    """
    prompt = f"""
    You are an expert in compliance reporting. Evaluate the following report based on its accuracy,
    completeness, and clarity in summarizing the required data points (e.g.,  Anti-Corruption and Anti-Bribery, 
    Lobbying and Political Contributions, Additional Governance Disclosures). Provide only a numerical score between 1 and 10 where 10 indicates an excellent report.
    

    Report:
    {report}

    Score (only the number):
    """
    score_str = judge_llm.invoke(prompt)
    try:
        # Extract the first number from the response
        score = float(score_str.strip().split()[0])
    except Exception as e:
        print("Error parsing score:", e)
        score = 0.0
    return score

with open('reports.json', 'r') as f:
    report = json.load(f)

gemini_report = report['gemini']
gpt_report = report['gpt']

# Evaluate both reports using the Gemini judge
gemini_score = evaluate_report(gemini_report, gemini_judge)
gpt_score = evaluate_report(gpt_report, gemini_judge)

print(f"Gemini Report Score: {gemini_score}")
print(f"GPT Report Score: {gpt_score}")

# Choose the best model's report based on the score
if gemini_score >= gpt_score:
    best_model = "Gemini"
    best_report = gemini_report
else:
    best_model = "GPT"
    best_report = gpt_report

print(f"\nBest Report is from {best_model} model.")
print("\nShortened Best Report:")
print(textwrap.shorten(best_report, width=2000, placeholder="..."))

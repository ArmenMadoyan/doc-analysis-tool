from dotenv import load_dotenv
import json
import os
import textwrap
from langchain.llms.base import LLM
from langchain_google_genai import GoogleGenerativeAI
import config
import parser
import chained
from importlib import reload

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

def prompt_and_create_env():
    env_file = ".env"
    if not os.path.exists(env_file):
        print("No .env file found. Please provide the following configuration:")
        file_dir = input("Provide folder absolute path: ").strip()
        tesseract_dir = input("Provide Tesseract absolute path: ").strip()
        openai_key = input("Provide ChatGPT (OpenAI) key: ").strip()
        gemini_key = input("Provide Gemini key: ").strip()

        # Create the .env file
        with open(env_file, "w", encoding="utf-8") as f:
            f.write(f"FILE_DIR={file_dir}\n")
            f.write(f"TESSERACT_DIR={tesseract_dir}\n")
            f.write(f"OPENAI_API_KEY={openai_key}\n")
            f.write(f"GEMINI_API_KEY={gemini_key}\n")
        print(".env file created.")
    else:
        load_dotenv(env_file)

if __name__ == "__main__":
    # Step 1: Prompt and create .env if needed
    load_dotenv()

    prompt_and_create_env()

    # Reload config to reflect new .env variables

    reload(config)

    print("Configuration Loaded:")
    print("FILE_DIR:", config.FILE_DIR)
    print("TESSERACT_DIR:", config.TESSERACT_DIR)
    print("OPENAI_API_KEY:", "****" if config.OPENAI_API_KEY else None)
    print("GEMINI_API_KEY:", "****" if config.GEMINI_API_KEY else None)

    # Step 2: Run Parser.py
    print("\nRunning Parser.py...")

    parser.main()

    # Step 3: Run chained.py
    print("\nRunning chained.py...")

    chained.main()

    # Instantiate the Gemini judge
    gemini_judge = chained.GeminiJudge()

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

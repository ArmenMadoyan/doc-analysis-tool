# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from sentence_transformers import SentenceTransformer
import chromadb
import openai
import numpy as np
from dotenv import load_dotenv
import os
from parser import Parser
load_dotenv()

ps = Parser()
path = "./chroma"
collection_name = "policy_embeddings"
model = SentenceTransformer("all-MiniLM-L6-v2")

# Run only once to store the data
# ps.parse_folder('./PoliciesForTheTask',chroma_path=path, collection_name=collection_name, parser_model=model)

client = openai.OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),  # This is the default and can be omitted
)

chroma_db = chromadb.PersistentClient(path=path)
chroma_collection = chroma_db.get_or_create_collection(collection_name)

# Define policy categories & associated key data points
categories = {
    "Anti-Corruption and Anti-Bribery": [
        "anti-corruption", "bribery", "fraud prevention", "whistleblowing",
        "due diligence", "sanctions", "legal cases"
    ],
    "Ethical Business Practices": [
        "code of conduct", "compliance system", "ethical decision-making",
        "non-compliance incidents"
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

# Query ChromaDB for each category
retrieved_data = {}

for category, keywords in categories.items():
    category_texts = []
    for keyword in keywords:
        query_embedding = model.encode(keyword).tolist()
        results = chroma_collection.query(query_embeddings=[query_embedding], n_results=3)  # Get top 3 matches

        for match in results["documents"][0]:  # Extract matched documents
            if match not in category_texts:
                category_texts.append(match)

    if category_texts:
        retrieved_data[category] = " ".join(category_texts)  # Combine retrieved text

# Notify about missing categories
missing_categories = [category for category in categories if category not in retrieved_data]
if missing_categories:
    print(f"Missing categories: {missing_categories}")
# print(retrieved_data)

def generate_report(retrieved_data):
    prompt = "Generate a structured policy narrative report based on the following extracted information:\n\n"

    for category, text in retrieved_data.items():
        prompt += f"## {category}\n{text}\n\n"

    prompt += "Ensure the report is well-structured and follows a formal policy report style."

    print(prompt)
    response = client.chat.completions.create(
        model="gpt-4-turbo",  # More cost-effective than GPT-4
        messages=[
            {"role": "system", "content": "You are an expert in corporate compliance writing structured reports."},
            {"role": "user", "content": prompt}],
        temperature=0.3  # Ensures factual consistency
    )
    # print(response)
    # return response
    return response.choices[0].message.content



# Generate and print the report
# report = generate_report(retrieved_data)
# print(report)

# print("# Zeno Consult Anti-Corruption Policy Narrative Report\n\n## Executive Summary\nThis report synthesizes the key components of Zeno Consult's Anti-Corruption Policy, emphasizing the firm's commitment to transparency, integrity, and zero tolerance towards corruption. The policy outlines the scope, definitions, and manifestations of corruption, as well as the mechanisms in place for prevention, reporting, and dealing with violations. This structured narrative aims to provide a comprehensive understanding of the policy's intent, implementation strategies, and the consequences of non-compliance.\n\n## 1. Introduction\nZeno Consult is committed to upholding the highest standards of honesty and ethical conduct. The Anti-Corruption Policy is a testament to our commitment to operate transparently and without tolerance for corruption in any form. This policy is fundamental in maintaining trust and integrity in all our business operations and relationships.\n\n## 2. Scope of the Policy\nThe Anti-Corruption Policy applies universally across the organization, covering all employees, whether temporary or permanent, as well as consultants, contractors, and other individuals or entities associated with Zeno Consult. This broad applicability ensures that all parties representing or interacting with Zeno are aligned with our ethical standards.\n\n## 3. Definitions and Forms of Corruption\nCorruption at Zeno is defined as the misuse of entrusted power for private gain, which includes, but is not limited to:\n- **Bribery:** Direct or indirect offers, gifts, or promises of value to influence the actions of others.\n- **Conflicts of Interest:** Situations where personal, financial, or other considerations have the potential to compromise or bias professional judgment and objectivity.\n- **Gifts and Invitations:** Any gifts or hospitality received or offered that might influence or appear to influence the impartiality of business decisions.\n- **Loans, Sponsoring, and Funding:** Financial transactions that could lead to dependencies or conflicts of interest.\n- **Lobbyism:** Engaging with policymakers in a manner that lacks transparency or attempts to gain undue advantage.\n\n## 4. Zero Tolerance Policy\nZeno enforces a strict zero-tolerance approach to corruption. Employees must not leverage their positions to offer, solicit, or accept improper benefits. All transactions and interactions must be conducted transparently and ethically, adhering to both legal standards and social norms.\n\n## 5. Detection and Reporting Mechanisms\n### 5.1 Alarm Signals\nEmployees are trained to recognize and report potential corruption indicators, such as unusual financial transactions, lack of documentation, or unexpected changes in vendor or employee behavior.\n### 5.2 Whistleblowing System\nA robust whistleblowing system is in place, allowing for anonymous or attributed reporting of suspicious activities, ensuring that all reports are handled with confidentiality and without retaliation.\n### 5.3 Training Programs\nRegular training sessions are conducted to reinforce awareness and understanding of the anti-corruption measures and reporting procedures.\n\n## 6. Consequences of Policy Violation\n### 6.1 For Zeno Consult\nViolations can severely damage Zeno's reputation, lead to legal penalties, financial losses, and disrupt business operations.\n### 6.2 For Individuals\nEmployees found engaging in corrupt activities face severe consequences including termination, legal action, and reputational damage.\n\n## 7. Policy Acceptance and Compliance\nAll employees and associated persons are required to explicitly acknowledge and comply with the Anti-Corruption Policy. Doubts concerning the propriety of any action or benefit should be immediately addressed with supervisors or through the designated reporting channels.\n\n## Conclusion\nZeno Consult's Anti-Corruption Policy is a cornerstone of our ethical framework, designed to foster a culture of integrity and transparency. Through comprehensive scope, clear definitions, strict enforcement, and robust reporting mechanisms, the policy ensures that all actions conducted under the banner of Zeno Consult adhere to the highest ethical standards. This structured approach not only protects the organization but also upholds its reputation and values in the global marketplace.")


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

response = generator("Summarize the following governance policies: " + str(retrieved_data), max_length=1000)
print(response[0]["generated_text"])
# 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
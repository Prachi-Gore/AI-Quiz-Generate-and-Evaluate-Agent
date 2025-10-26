from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain

load_dotenv()
llm = ChatOpenAI(
    model="gpt-4o-mini",  # cheaper model suitable for free trial
    temperature=0.6, # creativeness moderate
    api_key=os.getenv("OPENAI_API_KEY"),
)
# --- Quiz Generator ---
quiz_prompt = ChatPromptTemplate.from_template("""
You are a quiz master.
Generate 5 thought-provoking multiple-choice questions (MCQs)
from the following book summary.

Book Summary:
{context}

Return output strictly in JSON format like this:
[
  {{"question": "...", "options": ["A", "B", "C", "D"], "correct": "A"}},
  ...
]
""")

# quiz_chain = quiz_prompt | llm
quiz_chain = LLMChain(llm=llm, prompt=quiz_prompt)


# --- Answer Evaluator ---
evaluation_prompt = ChatPromptTemplate.from_template("""
You are a strict quiz evaluator.
Compare the user's answers with the correct answers and count how many are correct.

Correct Answers:
{questions}

User Answers:
{user_answers}

Return only JSON:
{{
  "score": int
}}
""")

evaluation_chain = LLMChain(llm=llm, prompt=evaluation_prompt)
summary_prompt = ChatPromptTemplate.from_template(
    "Write a 100 lines of summary for the book '{title}' by {author}."
)
# summary_chain = summary_prompt | llm
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# Prompt template for feedback
feedback_prompt = ChatPromptTemplate.from_template("""
You are an intelligent book mentor.
Here’s a summary of the book and the user’s quiz performance.

Book Context:
{relevant_summary}

User’s Score: {score}
Correct Answers:
{quiz}
User Answers:
{user_answers}

Based on the mistakes, provide **short, concise, and actionable feedback**.
Mention only the chapters, sections, or concepts the user should re-read. 
Be kind and precise. Limit the response to **1-3 sentences**.

Return JSON:
{{
  "feedback": str
}}
""")

feedback_chain = LLMChain(llm=llm, prompt=feedback_prompt)
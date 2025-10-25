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
You are an evaluator.
Evaluate user's answers based on these quiz questions and correct answers.

Questions:
{questions}

User Answers:
{user_answers}

Give score out of 5 and a feedback summary (pros & cons).
Return response as:
{{"score": int, "feedback": "string"}}
""")

evaluation_chain = LLMChain(llm=llm, prompt=evaluation_prompt)
summary_prompt = ChatPromptTemplate.from_template(
    "Write a 100 lines of summary for the book '{title}' by {author}."
)
# summary_chain = summary_prompt | llm
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# Prompt template for feedback
feedback_prompt = ChatPromptTemplate.from_template("""
You are an educational assistant. A user has answered a quiz with the following results:

Quiz Questions:
{quiz}

Evaluation / Mistakes:
{evaluation}

The user scored {score} out of 5.

Instruction:
If the score is below 3, suggest specific pages or sections of the book they should reread to improve their understanding.
Provide concise, clear, and helpful advice. If the score is high, give a positive encouragement message.
""")

feedback_chain = LLMChain(llm=llm, prompt=feedback_prompt)
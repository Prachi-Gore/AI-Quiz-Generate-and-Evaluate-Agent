from langgraph.graph import StateGraph
# from langgraph.graph.message import add_messages
from app.prompts import  evaluation_chain,feedback_chain
from typing import TypedDict, List,Optional

class QuizState(TypedDict):
    # book_id: str
    quiz: str
    user_answers: List[str]
    evaluation: Optional[str]
    score: Optional[float]
    feedback: Optional[str]

# --- Nodes ---
# class QuizNode:
#     def run(self, state: dict):
#         # Get book summary
#         book_id = state.get("book_id")
#         if not book_id:
#             raise ValueError("book_id is missing in state")

#         summary = get_relevant_summary(book_id)

#         # Run the quiz generation pipeline
#         quiz = quiz_chain.ainvoke({"context": summary})

#         return {"quiz": quiz}


class EvaluationNode:
    async def run(self, state: QuizState):
        evaluation_result = await evaluation_chain.ainvoke({
            "questions": state["quiz"],
            "user_answers": state["user_answers"]
        })

        score = evaluation_result.get("score", 0)
        evaluation_text = evaluation_result.get("feedback", "")
        return {"evaluation": evaluation_text, "score": score}


class FeedbackNode:
    async def run(self, state: QuizState):
        score = state.get("score", 0)
        quiz = state.get("quiz", "")

        if score < 3:
            feedback_text = await feedback_chain.ainvoke({
                "quiz": quiz,
                "evaluation": state["evaluation"],
                "score":score,
                "instruction": (
                    "Based on user's mistakes, suggest pages or sections "
                    "from the quiz or book they should reread."
                )
            })
        else:
            feedback_text = "Great job! You’ve answered most questions correctly."

        return {"feedback": feedback_text}

def evaluate_quiz_graph():
    # Create a workflow graph for the quiz
    workflow = StateGraph(QuizState)

    # Add nodes with the updated run methods
    workflow.add_node("EvaluationNode", EvaluationNode().run)
    workflow.add_node("FeedbackNode", FeedbackNode().run)

    workflow.add_edge("EvaluationNode", "FeedbackNode")

    workflow.set_entry_point("EvaluationNode")
    workflow.set_finish_point("FeedbackNode")

    # Compile the graph to make it executable
    compiled_workflow = workflow.compile()

    return compiled_workflow
import os
from dotenv import load_dotenv
from groq import Groq
from retrieval import get_relevant_documents

load_dotenv()

MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0
chat_history = []

def ask_question(user_question):
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        return "❌ GROQ_API_KEY not found in environment variables."

    client = Groq(api_key=groq_key)

    # Step 1: Rewrite follow-up question into standalone if history exists
    if chat_history:
        history_text = "\n".join([f"User: {h['user']}\nBot: {h['bot']}" for h in chat_history])
        rewrite_prompt = (
            "You are an assistant that rewrites follow-up questions into standalone questions.\n"
            f"Conversation history:\n{history_text}\n"
            f"New question: {user_question}\n"
            "Return ONLY the rewritten standalone question."
        )
        rewrite_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=TEMPERATURE
        )
        search_question = rewrite_response.choices[0].message.content.strip()
    else:
        search_question = user_question

    # Step 2: Retrieve relevant documents
    docs = get_relevant_documents(search_question, k=3)

    if not docs:
        answer = "I don't have enough information to answer that question based on the provided documents."
        chat_history.append({"user": user_question, "bot": answer})
        return answer

    # Step 3: Build context
    context = "\n\n".join([chunk["content"] for chunk in docs])
    prompt = (
        "You are a strict RAG assistant.\n"
        "Answer ONLY using the provided context. If the answer is not present, say:\n"
        "'I don't have enough information based on the provided documents.'\n\n"
        f"Context:\n{context}\n\nQuestion:\n{user_question}"
    )

    # Step 4: Generate answer
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE
    )

    answer = response.choices[0].message.content.strip()
    chat_history.append({"user": user_question, "bot": answer})
    return answer

def start_chat():
    print("Welcome! Ask me anything. Type 'quit' to exit.")
    while True:
        question = input("\nYour question: ")
        if question.lower() == "quit":
            print("Goodbye!")
            break
        print(f"Bot: {ask_question(question)}")

if __name__ == "__main__":
    start_chat()

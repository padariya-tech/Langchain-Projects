from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    provider="auto",
    max_new_tokens=256,
)

model = ChatHuggingFace(llm=llm)

# print(model.invoke("What is the capital of India?").content)
# chatbot is without context
# we need to store the history of the chat
messages =[
    SystemMessage(content="You are a helpful assistant."),
]
# chat_history = []
while True:
    user_input = input("You: ")
    # chat_history.append({"role": "user", "content": user_input})
    messages.append(HumanMessage(content=user_input))
    print("             ")
    if user_input == "EXIT":
        break

    result = model.invoke(messages)
    messages.append(AIMessage(content=result.content))
    print("AI Assistant:", result.content)
    # chat_history.append({"role": "assistant", "content": result.content})
    print("             ")

# print("Chat History: ", chat_history)

print(messages)




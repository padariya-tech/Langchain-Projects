from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    provider="auto",
    max_new_tokens=256,
)

model = ChatHuggingFace(llm=llm)

chat_template = ChatPromptTemplate([
    ('system','You are a helpful {domain} assistant.'),
    ('human','explain in simple terms: {question}')
])

prompt = chat_template.invoke({'domain': 'AI', 'question': 'cricket'})

print(prompt)
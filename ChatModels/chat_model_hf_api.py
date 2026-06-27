from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    provider="auto",
    max_new_tokens=256,
)

chat = ChatHuggingFace(llm=llm)

response = chat.invoke(
    [HumanMessage(content="What is the capital of India?")]
)

print(response.content)
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    provider="auto",
    max_new_tokens=256,
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    input_variables=["text"],
    template="Please summarize the following text:\n{text}",
    output_parser=StrOutputParser(),
)

result = model.invoke(prompt.invoke({"text": "hello world"}))

print(result.content)

loader = TextLoader(file_path="a.txt", encoding="utf-8")
documents = loader.load()

print(documents)

print(documents[0].page_content)

print(documents[0].metadata)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"text": documents[0].page_content})

print(result)
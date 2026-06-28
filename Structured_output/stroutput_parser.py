from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    provider="auto",
    max_new_tokens=256,
)

model = ChatHuggingFace(llm=llm)

# 1st -> detailed report

template_1 = PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables=['topic']
)


template_2 = PromptTemplate(
    template='write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

# prompt1 = template_1.invoke({
#     'topic': 'Machine learning'
# })

# result = model.invoke(prompt1)
# print(result.content)

# prompt2 = template_2.invoke({
#     'text': result.content
# })

# result2 = model.invoke(prompt2)

# print(result2.content)

parser = StrOutputParser()
chain = template_1 | model | parser | template_2 | model | parser

result = chain.invoke({
    'topic': 'Machine learning'
})

print(result)
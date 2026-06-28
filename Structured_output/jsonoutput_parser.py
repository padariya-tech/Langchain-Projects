from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    provider="auto",
    max_new_tokens=256,
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()


template = PromptTemplate(
    template='give me the name , age and city of fictional person \n , {format_instructions}',
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# prompt = template.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# print(final_result)

chain = template | model | parser

result = chain.invoke({})

print(result)
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers  import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    provider="auto",
    max_new_tokens=256,
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(gt=18,description="The age of the person")
    city: str = Field(description="The city of the person")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='give me the name , age and city of fictional person \n , {format_instructions}',
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt = template.invoke({})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(type(final_result.age))
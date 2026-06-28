from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers  import StructuredOutputParser, ResponseSchema


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    provider="auto",
    max_new_tokens=256,
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact_1', description='The first fact about the topic'),
    ResponseSchema(name='fact_2', description='The second fact about the topic'),
    ResponseSchema(name='fact_3', description='The third fact about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 fact about {topic}, {format_instructions}',
    input_variables=['topic'],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt = template.invoke({
    'topic': 'Machine learning'
})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)
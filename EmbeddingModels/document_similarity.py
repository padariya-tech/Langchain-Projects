from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about virat'

docs_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

from sklearn.metrics.pairwise import cosine_similarity

scores = cosine_similarity([query_embedding], docs_embeddings)[0]


index,score = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[0] # Get the index of the most similar document and its score

print(query)
print(documents[index])
print(score)
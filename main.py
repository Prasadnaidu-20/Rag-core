from src.pdf_loader import load_pdf
from src.chunker import overLappingChunk
from src.embedder import embedder
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from src.gemini import llm_response

client = QdrantClient(":memory:")
print(client.get_collections())

text = load_pdf("data/RAG_Notes_Digital.pdf")
# print(text[:500])

overlapping_size = 100
chunk_size = 500
chunks = overLappingChunk(text,chunk_size,overlapping_size)

# print(len(chunks))
# print(chunks[0])
# print(chunks[1])
# print(chunks[2])

vectors = embedder(chunks)
if not client.collection_exists("rag_notes"):
    client.create_collection(
    collection_name="rag_notes",
    vectors_config={
        "size": vectors.shape[1],   #it is the dimension of the vector,
        "distance": "Cosine"
    }
)
#upsert is a combination of update and insert, it will insert the data if it does not exist, and update it if it does exists
client.upsert(
    collection_name="rag_notes",
    points=[
        PointStruct(        #pointStruct is a class that represents a point in the vector space, it has an id, a vector and a payload which is a dictionary that can store any additional information we want to assoctiate with the point
            id = i,
            vector = vectors[i].tolist(),  #vectors[i] is a numpy array, we need to convert it to a list before inserting it into the database
            payload = {
                "text": chunks[i],
                "source": "RAG_Notes_Digital.pdf",
                "chunk_index": i
            }
        )
        for i in range(len(chunks)) 
    ]
)

# Now we have inserted the chunks and their corresponding vectors into the Qdrant database. We can now perform similarity search on the database to retrieve the most relevant chunks based on a query.

queries = [
"How does the model deal with new information?"
]

for query in queries:
    query_vector = embedder([query])[0]  #here also we need to pass a list to the embedder function, so we need to convert the query string to a list before passing it to the embedder function , why [0] bec

    #we are searching for the most similar vectors to the query vector in the db, and we are retrieving the top 3 most similar vectors
    results = client.query_points(   
        collection_name = "rag_notes",
        query = query_vector.tolist(),
        limit = 3
    )

    context = "\n\n".join(
    point.payload["text"]
    for point in results.points
    )

    prompt = f"""       
        You are a helpful assistant.

        Answer ONLY from the provided context.

        If the answer is not present in the context,
        say:

        'I could not find that information in the documents.'

        Context:
        {context}

        Question:
        {query}
        """
    
    response = llm_response(prompt)
    print(response)


# print(vectors.shape)
# print(vectors[0])
# print(vectors[1])
# print(vectors[2])
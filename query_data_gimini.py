import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

CHROMA_PATH = "chroma_gimini"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    query_text = 'Who are the animals that Alice met?'



    embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    #vector = embedding_function.embed_query(query_text)
    #print(len(vector))

    # Prepare the DB.
  
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=10)
    relevance_scores = [result[1] for result in results]

# Print the relevance scores
    print(relevance_scores)
    
    if len(results) == 0:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    prompt = prompt_template.format(context=context_text, question=query_text)
    #print(prompt)

    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(prompt)
    response_text = response.text

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
if __name__ == "__main__":
    main()

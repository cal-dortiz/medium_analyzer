from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == '__main__':
    print('ingestion') 

    loader = TextLoader(("/Users/danielortiz/workspace/github.com/cal-dortiz/medium_analyzer/mediumblog1.txt"))
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(texts)

    embedings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    PineconeVectorStore.from_documents(texts, embedings, index_name=os.environ('INDEX_NAME'))
    print("finish")
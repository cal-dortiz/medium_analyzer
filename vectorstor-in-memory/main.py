import os
from venv import create
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub


if __name__ == '__main__':
    print('hi')
    pdf_path = '/Users/danielortiz/workspace/github.com/cal-dortiz/medium_analyzer/vectorstor-in-memory/2210.03629v3.pdf'
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator = "\n")
    docs=text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()
    vectorstor = FAISS.from_documents(docs, embeddings)
    vectorstor.save_local("faiss_index_react") # save local

    new_vectorstor = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    retreieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        OpenAI(), retreieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(
        new_vectorstor.as_retriever(), combine_docs_chain
    )

    res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
    print(res["answer"])

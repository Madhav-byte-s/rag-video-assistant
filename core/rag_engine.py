import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from core import vector_store
from core.vector_store import build_vector_store, load_vector_store, get_retiever

def get_llm():
    return ChatMistralAI(
        model="ministral-8b-2512",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )

def format_docs(docs):
    formatted_docs = []
    
    return "\n\n".join([doc.page_content for doc in docs])

def build_rag_chain(transcript:str):
    vector_store = build_vector_store(transcript)
    retriever = get_retiever(vector_store, k=4)

    llm =get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer. Use the provided meeting transcript to answer the user's question. "
                "If the answer is not present in the transcript, respond with 'I don't know'.",
            ),
            ("human", "{context}\n\nQuestion: {question}"),
        ]
    )

    #full LCEL RAG pipeline
    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retiever(vector_store, k=4)

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer. Use the provided meeting transcript to answer the user's question. "
                "If the answer is not present in the transcript, respond with 'I don't know'.",
            ),
            ("human", "{context}\n\nQuestion: {question}"),
        ]
    )

    #full LCEL RAG pipeline
    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def ask_question(rag_chain, question: str) -> str:
    print(f"Question: {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")
    return answer

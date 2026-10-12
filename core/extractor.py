#Actionableitems,decision,questions

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import os

def get_llm():
    return ChatMistralAI(
        model="ministral-8b-2512",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2,
    )


def build_chain(system_prompt :str):
     llm = get_llm()
     return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{text}")]
        )
        | llm 
        | StrOutputParser()
    )
def extract_action_items(transcript:str) -> str:
    system_prompt = "You are an expert meeting summarizer. Extract all actionable items from this meeting transcript and present them in a clear and concise bullet point format."
    chain = build_chain(system_prompt)
    return chain.invoke(transcript)


def extract_key_decisions(transcript:str) -> str:
    system_prompt = "You are an expert meeting summarizer. Extract all key decisions made during this meeting from the transcript and present them in a clear and concise bullet point format."
    chain = build_chain(system_prompt)
    return chain.invoke(transcript)

def extract_questions(transcript:str) -> str:
    system_prompt = "You are an expert meeting summarizer. Extract all questions raised during this meeting from the transcript and present them in a clear and concise bullet point format."
    chain = build_chain(system_prompt)
    return chain.invoke(transcript)
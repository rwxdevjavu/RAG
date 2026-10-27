from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from src.core.gemini_llm import get_llm
from src.rag.retriever import get_retriever

llm = get_llm()

prompt = ChatPromptTemplate.from_template(
    """
Answer the question based ONLY on the context below.
If the answer is not in the context, say "I don't know."

Context: {context}

Question: {question}
"""
)


def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)


def retriever_and_format(query):
    r = get_retriever()
    docs = r.invoke(query)
    return format_docs(docs)


def get_rag_chain():
    return (
        {"context": RunnableLambda(retriever_and_format), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


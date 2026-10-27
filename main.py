from src.rag.chain import get_rag_chain
rag = get_rag_chain()
answer = rag.invoke("does it support SSO SAML")
print(answer)



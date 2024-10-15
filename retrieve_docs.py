from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
import os

# 모델 선정
model = ChatOpenAI(model="gpt-4o-mini")

# txt 파일 불러와서 document load
loader = TextLoader("oasis.txt", encoding="utf-8")
documents = loader.load()

# 너무 큰 텍스트는 처리하기 어러워 document를 분할
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# document Chunk들을 벡터로 변환하기 위해 OpenAI의 임베딩 모델 사용
embeddings = OpenAIEmbeddings()

# FAISS를 사용하여 벡터들에 대한 데이터베이스를 만들어 효과적으로 검색할 수 있게 사용
db = FAISS.from_documents(docs, embeddings)


# 사용자 질문에 맞는 document를 검색해서 반환
def retrieve_docs(query: str):
    retriever = db.as_retriever()
    docs = retriever.invoke(query)
    return docs[0].page_content

# 검색된 document의 내용을 반환
def get_var(question, *args, **kwargs):
    context = retrieve_docs(question)
    return {"output": context}
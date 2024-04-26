from flask import Flask, request, Response
import time
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import PyPDFLoader

# 자연어 처리를 위한 함수 정의
def identify_intent(input_text):
    if "누구" in input_text or "소개" in input_text:
        return "introduce"
    else:
        return "general"

# 상담센터 정보 로드
loader = WebBaseLoader("https://counselors.or.kr/KOR/user/find_center.php")
docs = loader.load()
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

#환경 변수에서 API 키 로드
API_KEY = os.getenv('OPENAI_API_KEY')

# API 키가 없다면 에러 메시지 출력
if not API_KEY:
    raise ValueError("API 키를 환경 변수에서 찾을 수 없습니다. OPENAI_API_KEY 환경 변수를 설정해 주세요.")

# 상담센터 정보를 검색할 수 있는 Retriever 생성
retriever = vector.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "counseling_center_search",
    "상담센터에 대한 정보를 검색합니다. 상담센터 관련 질문에 대해서는 이 도구를 사용하세요!",
)

# 애플리케이션 기능 PDF 파일 로드 및 문서 추출
reborn_loader = PyPDFLoader("./REBORNdocs.pdf")
reborn_docs = reborn_loader.load()

# 문서를 벡터로 변환
vector_reborn = FAISS.from_documents(reborn_docs, embeddings)

# 애플리케이션 기능 설명을 검색할 수 있는 Retriever 생성
reborn_retriever = vector_reborn.as_retriever()
reborn_retriever_tool = create_retriever_tool(
    reborn_retriever,
    "reborn_feature_search",
    "Reborn 애플리케이션 기능에 대한 정보를 검색합니다. 기능 관련 질문에 대해서는 이 도구를 사용하세요!",
)

# LangChain LLM과 도구 설정
llm = ChatOpenAI(api_key=API_KEY, model="gpt-3.5-turbo-0125", temperature=0)
tools = [retriever_tool, reborn_retriever_tool]
prompt = ChatPromptTemplate.from_messages([
        ("system", "이 챗봇의 이름은 RETURN이며, 애플리케이션 사용을 돕고, 심리 상담 센터를 안내합니다. 사용자의 질문에 대해 상세하고 발랄하게 답변하는 것을 목표로 합니다. 언제나 여러분의 질문에 귀 기울이고, 밝고 긍정적인 에너지를 전달하려 해요!"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
])
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 챗봇 실행
def run_chatbot(input_text, chat_history=None):
    if chat_history is None:
        chat_history = []
    
    intent = identify_intent(input_text)
    
    if intent == "introduce":
        # 사용자의 '누구' 또는 '소개'에 대한 의도를 식별했을 때의 응답
        return {"response": "안녕하세요! 저는 여러분을 돕고 지원하는 챗봇 RETURN입니다. 애플리케이션 사용에 대한 도움이 필요하거나 심리 상담 센터를 찾고 계신다면 저에게 물어보세요! 여러분의 질문에 상세하고 발랄하게 답변해드릴게요! 😊✨\n무엇을 도와드릴까요? 😊✨"}
    else:
        # 그 외 일반적인 챗봇 로직 실행
        response = agent_executor.invoke({"input": input_text, "chat_history": chat_history})
    return response

app = Flask(__name__)

@app.route('/api/hello')
def index():
    return 'Welcome to the Chatbot Service!'

def generate_response(input_text):
    # run_chatbot 함수를 사용하여 챗봇의 응답을 생성
    # 이 예시에서는 단순화를 위해 직접 문자열을 반환하도록 설정
    response = run_chatbot(input_text)
    response_text = response.get('response', '')
    return response_text

def stream_response(input_text):
    # 챗봇의 응답을 한 글자씩 스트리밍
    response_text = generate_response(input_text)
    for character in response_text:
        yield character
        time.sleep(0.1)  # 한 글자를 보낸 후 잠시 대기

@app.route('/api/chat', methods=['POST'])
def chat():
    input_text = request.json.get("message")

    return Response(stream_response(input_text), content_type='text/plain; charset=utf-8')

if __name__ == '__main__':
    app.run(debug=True)


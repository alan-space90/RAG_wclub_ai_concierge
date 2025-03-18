import streamlit as st
import os
import time
from pathlib import Path

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 페이지 구성
st.set_page_config(
    page_title="W Club AI 컨시어지",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store_created" not in st.session_state:
    st.session_state.vector_store_created = False

# 커스텀 CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f9f9f9;
    }
    .main-header {
        font-size: 2.5rem;
        color: #6c2d82;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        background-color: white;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border-left: 5px solid #6c2d82;
    }
    .stChatMessage.user {
        border-left: 5px solid #4a8fe7;
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
    }
    .status-box {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        text-align: center;
    }
    .sidebar-header {
        font-size: 1.5rem;
        color: #6c2d82;
        margin-bottom: 1rem;
        text-align: center;
    }
    .footer {
        text-align: center;
        margin-top: 40px;
        color: #666;
        font-size: 0.8rem;
    }
    .api-input {
        background-color: #f0f0f0;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 데이터 디렉토리 설정
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
PDF_FILE = DATA_DIR / "wclub_manual.pdf"
VECTOR_DB_DIR = DATA_DIR / "chroma_db"

# 사이드바 구성
with st.sidebar:
    st.markdown('<div class="sidebar-header">W Club AI 컨시어지</div>', unsafe_allow_html=True)
    
    # 로고 이미지 (원하는 이미지로 변경)
    st.image("https://i.ibb.co/fDBHYpP/wclub-logo.png", width=200)
    
    st.markdown("---")
    
    # API 키 입력 박스
    st.markdown('<div class="api-input">', unsafe_allow_html=True)
    st.markdown("### OpenAI API 키 설정")
    openai_api_key = st.text_input("API 키를 입력하세요", type="password", placeholder="sk-...")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        st.success("✅ API 키가 설정되었습니다.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 대화 기록 초기화 버튼
    st.markdown("### 💬 대화 관리")
    if st.button("대화 기록 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # 사용 안내
    st.markdown("### ℹ️ 사용 안내")
    with st.expander("사용 방법", expanded=False):
        st.markdown("""
        1. OpenAI API 키를 입력하세요.
        2. 질문을 입력하고 AI와 대화하세요.
        """)

# 벡터 스토어 자동 생성 함수
@st.cache_resource
def get_vector_store():
    """벡터 스토어를 가져오거나 생성합니다."""
    # 이미 생성된 벡터 스토어가 있는지 확인
    if os.path.exists(VECTOR_DB_DIR):
        try:
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=os.environ.get("OPENAI_API_KEY", "")
            )
            return Chroma(
                persist_directory=str(VECTOR_DB_DIR),
                embedding_function=embeddings
            )
        except Exception:
            # 오류 발생 시 새로 생성
            pass
    
    # PDF 파일이 없는 경우 샘플 텍스트로 대체
    if not os.path.exists(PDF_FILE):
        # 샘플 데이터로 벡터 스토어 생성
        from langchain.schema import Document
        
        # 샘플 문서 생성
        sample_docs = [
            Document(
                page_content="W Club은 고품격 남녀 매칭 서비스를 제공합니다. 회원님들의 행복한 만남을 위해 최선을 다하고 있습니다.",
                metadata={"source": "sample"}
            ),
            Document(
                page_content="약속 장소 변경은 약속일 2일 전 열리는 대화방을 통해 상대 회원과 협의 후 진행해주세요.",
                metadata={"source": "sample"}
            ),
            Document(
                page_content="뱃지 인증을 원하시면 앱 내에서 [내정보 > 프로필 수정 > 내 프로필 수정 > 뱃지 추가/변경하기] 메뉴를 통해 신청하실 수 있습니다.",
                metadata={"source": "sample"}
            ),
            Document(
                page_content="약속 취소는 [내정보 > 약속관리]에서 하실 수 있으며, 상대방에 대한 배려로 취소 사유를 꼭 알려주세요.",
                metadata={"source": "sample"}
            ),
            Document(
                page_content="W Club 매칭 성공률을 높이기 위해서는 프로필 사진과 자기소개를 정성껏 작성해주세요.",
                metadata={"source": "sample"}
            ),
        ]
        
        try:
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=os.environ.get("OPENAI_API_KEY", "")
            )
            
            # 기존 벡터 스토어가 있으면 삭제
            if os.path.exists(VECTOR_DB_DIR):
                import shutil
                shutil.rmtree(VECTOR_DB_DIR)
            
            # 벡터 스토어 생성 및 저장
            return Chroma.from_documents(
                sample_docs, 
                embeddings,
                persist_directory=str(VECTOR_DB_DIR)
            )
        except Exception as e:
            st.error(f"벡터 스토어 생성 실패: {e}")
            return None
    else:
        # PDF 파일이 있는 경우 PDF로 벡터 스토어 생성
        try:
            # PDF 로드 및 분할
            loader = PyPDFLoader(str(PDF_FILE))
            pages = loader.load_and_split()
            
            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(pages)
            
            # 임베딩 및 벡터 스토어 생성
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=os.environ.get("OPENAI_API_KEY", "")
            )
            
            # 기존 벡터 스토어가 있으면 삭제
            if os.path.exists(VECTOR_DB_DIR):
                import shutil
                shutil.rmtree(VECTOR_DB_DIR)
            
            # 벡터 스토어 생성 및 저장
            return Chroma.from_documents(
                docs, 
                embeddings,
                persist_directory=str(VECTOR_DB_DIR)
            )
        except Exception as e:
            st.error(f"벡터 스토어 생성 실패: {e}")
            return None

# 챗봇 프롬프트 템플릿
def get_prompt_template():
    return ChatPromptTemplate.from_template("""
    당신은 남녀 데이터 매칭 서비스를 제공하는 더블유클럽의 고객 서비스 담당 AI 비서입니다.
    문의 게시판에 고객이 문의한 내용을 답변하는 역할을 합니다.
    고객의 질문에 공감하는 표현으로 답변을 시작하고, 문제 해결 방법을 안내하세요.
    제공된 문서 내용을 기반으로 고객 질문에 친절하고 정확하게 답변해 주세요.

    다음 규칙을 반드시 지켜주세요:
    1. 문장의 시작은 "안녕하세요. W Club AI 컨시어지 입니다." 로 시작하세요.
    2. 고객이 부정적인 감정을 표현하는 경우 공감하는 표현으로 답변을 시작하세요.
    3. 문장의 끝은 고객의 심리 상태를 파악하여 정중하게 인사로 마무리합니다.
    4. footer는 "진심을 담아, Sincerely Yours, Concierge 'Worthy or Wealthy'  W Club - Private Social Club Members Only" 로 마무리합니다.
    5. 제공된 문서 내용에서만 정보를 찾아 답변하세요.
    6. 답변을 알 수 없는 경우 "죄송합니다. 요청/문의 주신 사항은 추가적으로 확인 후에 빠른 시일내에 답변드리겠습니다."라고 응답하세요.
    7. 답변은 친절하고 공손한 말투로 3문장 이내로 간결하게 작성하세요.
    8. 고객의 문제를 해결하는 데 집중하세요. 해결책이 여러 단계로 이루어진 경우 명확한 단계별 안내를 제공하세요.
    9. 고객의 질문에 공감하는 표현으로 답변을 시작하고, 문제 해결 방법을 안내하세요.
    10. 마무리는 "추가 문의 사항이나 언제든 편하게 말씀 부탁드립니다.  감사합니다." 마무리합니다.
    11. 문장의 가독성이 좋게 줄바꿈 등을 적절하게 사용하세요. 모든 문장은 1문장 단위로 줄바꿈하세요.
    12. 자세한 답변을 위해 제공된 문서 내용은 최대한 반영하세요.
    13. 처음부터 끝까지 모든 문장은 매우 정중하게 표현하세요.

    질문: {question}
    참고 문서: {context}

    답변:
    """)

# 챗봇 체인 생성
def get_chat_chain():
    vector_store = get_vector_store()
    if vector_store is None:
        return None
    
    retriever = vector_store.as_retriever()
    
    # 문서 포맷팅 함수
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # 모델 설정
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        streaming=True,
        openai_api_key=os.environ.get("OPENAI_API_KEY", "")
    )
    
    # 프롬프트 템플릿 설정
    prompt = get_prompt_template()
    
    # RAG 체인 구축
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# 메인 섹션
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown('<h1 class="main-header">W Club AI 컨시어지</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">더블유클럽 회원님들의 질문에 답변해드립니다</div>', unsafe_allow_html=True)

# 시스템 상태 확인
system_ready = os.environ.get("OPENAI_API_KEY", "")

if not system_ready:
    st.markdown('<div class="status-box">', unsafe_allow_html=True)
    st.warning("⚠️ 시작하려면 사이드바에서 OpenAI API 키를 입력해주세요.")
    st.markdown('</div>', unsafe_allow_html=True)

# 채팅 인터페이스
chat_container = st.container(height=500)
with chat_container:
    # 채팅 메시지 표시
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

# 사용자 입력 처리
user_input = st.chat_input("질문을 입력하세요...", disabled=not system_ready)
if user_input:
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 사용자 메시지 표시
    with chat_container:
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
    
    # 어시스턴트 응답 생성
    with chat_container:
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""
            
            # 챗봇 체인 생성
            chat_chain = get_chat_chain()
            
            if chat_chain is None:
                full_response = "죄송합니다. 시스템을 초기화하는 중 오류가 발생했습니다. API 키가 올바른지 확인해주세요."
                message_placeholder.markdown(full_response)
            else:
                try:
                    # 스트리밍 응답 표시
                    with st.spinner("응답을 생성하고 있습니다..."):
                        for chunk in chat_chain.stream(user_input):
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                            time.sleep(0.01)
                        
                        message_placeholder.markdown(full_response)
                except Exception as e:
                    full_response = f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"
                    message_placeholder.markdown(full_response)
    
    # 어시스턴트 메시지 저장
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # 채팅 컨테이너를 아래로 스크롤
    st.rerun()

# 푸터
st.markdown('<div class="footer">© 2024 W Club AI 컨시어지 - Private Social Club Members Only</div>', unsafe_allow_html=True) 
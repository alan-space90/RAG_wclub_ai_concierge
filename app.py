import streamlit as st
import os
import time
import hashlib
import base64
import json
from pathlib import Path
from nacl.secret import SecretBox
from nacl.encoding import Base64Encoder

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
    layout="centered",
    initial_sidebar_state="expanded",
)

# 세션 상태 초기화
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store_created" not in st.session_state:
    st.session_state.vector_store_created = False

# 암호화 키 설정 (실제 앱에서는 환경 변수나 더 안전한 방법으로 관리해야 함)
raw_key = st.secrets.get("ENCRYPTION_KEY", "wclubsecretkey12")
# 키를 정확히 32바이트로 변환
ENCRYPTION_KEY = hashlib.sha256(raw_key.encode()).digest()
box = SecretBox(ENCRYPTION_KEY)

# OpenAI API 키 설정
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

# 데이터 디렉토리 설정
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
USERS_FILE = DATA_DIR / "users.enc"
PDF_FILE = DATA_DIR / "wclub_manual.pdf"
VECTOR_DB_DIR = DATA_DIR / "chroma_db"

# 암호화 및 복호화 함수
def encrypt_data(data):
    try:
        # 데이터를 JSON 문자열로 변환
        json_data = json.dumps(data)
        # 암호화
        encrypted = box.encrypt(json_data.encode('utf-8'), encoder=Base64Encoder)
        return encrypted.decode('utf-8')
    except Exception:
        return ""

def decrypt_data(encrypted_data):
    try:
        # 복호화
        decrypted = box.decrypt(encrypted_data.encode('utf-8'), encoder=Base64Encoder)
        # JSON으로 파싱
        return json.loads(decrypted.decode('utf-8'))
    except Exception:
        return {}

# 사용자 데이터 로드 및 저장 함수
def load_users():
    # 기본 사용자 정보 정의
    default_users = {
        "admin@wclub.com": {
            "password": hashlib.sha256("admin123".encode()).hexdigest(),
            "name": "관리자"
        },
        "guest@wclub.com": {
            "password": hashlib.sha256("guest123".encode()).hexdigest(),
            "name": "게스트"
        }
    }
    
    # 파일이 없으면 기본 사용자 생성
    if not USERS_FILE.exists():
        save_users(default_users)
        return default_users
    
    try:
        # 파일 읽기 시도
        with open(USERS_FILE, 'rb') as f:
            encrypted_data = f.read().decode('utf-8')
        
        # 복호화 시도
        users = decrypt_data(encrypted_data)
        
        # 복호화 실패하거나 사용자 정보가 비어있으면 기본 사용자 다시 생성
        if not users:
            save_users(default_users)
            return default_users
            
        return users
    except Exception:
        # 오류 발생 시 기본 사용자 생성
        save_users(default_users)
        return default_users

def save_users(users):
    encrypted_data = encrypt_data(users)
    with open(USERS_FILE, 'w') as f:
        f.write(encrypted_data)

# 로그인 검증 함수
def verify_login(email, password):
    users = load_users()
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    if email in users and users[email]["password"] == password_hash:
        return True, users[email]["name"]
    return False, ""

# 로그인 UI
def login_ui():
    st.markdown(
        """
        <style>
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background-color: #f8f9fa;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.image("https://i.ibb.co/fDBHYpP/wclub-logo.png", width=200)
    st.title("W Club AI 컨시어지")
    
    with st.container():
        st.markdown("<div class='login-container'>", unsafe_allow_html=True)
        
        email = st.text_input("이메일", placeholder="이메일을 입력하세요")
        password = st.text_input("비밀번호", type="password", placeholder="비밀번호를 입력하세요")
        
        if st.button("로그인", use_container_width=True):
            if email and password:
                success, user_name = verify_login(email, password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user_name = user_name
                    st.rerun()
                else:
                    st.error("이메일 또는 비밀번호가 올바르지 않습니다.")
            else:
                st.warning("이메일과 비밀번호를 모두 입력해주세요.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: gray; font-size: 0.8em;'>
            © 2023 W Club. All rights reserved.<br>
            Worthy or Wealthy - Private Social Club Members Only
            </div>
            """,
            unsafe_allow_html=True
        )

# PDF 파일 업로드 함수
def upload_pdf():
    st.title("PDF 매뉴얼 업로드")
    
    uploaded_file = st.file_uploader("W Club 매뉴얼 PDF를 업로드해주세요", type="pdf")
    
    if uploaded_file:
        with open(PDF_FILE, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success("PDF 파일이 성공적으로 업로드되었습니다.")
        
        # 벡터 스토어 생성
        create_vector_store()
        st.rerun()

# 벡터 스토어 생성 함수
@st.cache_resource
def get_vector_store():
    if not VECTOR_DB_DIR.exists() or not PDF_FILE.exists():
        return None
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    
    return Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings
    )

def create_vector_store():
    # OpenAI API 키 설정
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    # PDF 로드 및 분할
    loader = PyPDFLoader(str(PDF_FILE))
    pages = loader.load_and_split()
    
    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)
    
    # 임베딩 및 벡터 스토어 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 기존 벡터 스토어가 있으면 삭제
    if VECTOR_DB_DIR.exists():
        import shutil
        shutil.rmtree(VECTOR_DB_DIR)
    
    # 벡터 스토어 생성 및 저장
    Chroma.from_documents(
        docs, 
        embeddings,
        persist_directory=str(VECTOR_DB_DIR)
    )
    
    st.session_state.vector_store_created = True

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
        openai_api_key=OPENAI_API_KEY
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

# 챗봇 UI
def chatbot_ui():
    st.title(f"W Club AI 컨시어지")
    st.markdown(f"안녕하세요, {st.session_state.user_name}님! 무엇을 도와드릴까요?")
    
    # 사이드바
    with st.sidebar:
        st.image("https://i.ibb.co/fDBHYpP/wclub-logo.png", width=150)
        st.markdown("## W Club AI 컨시어지")
        st.markdown("더블유클럽 회원님들의 질문에 답변해드립니다.")
        
        if st.button("로그아웃"):
            st.session_state.clear()
            st.rerun()
        
        # 관리자만 PDF 업로드 가능
        if st.session_state.get("user_name") == "관리자":
            st.markdown("---")
            st.markdown("### 관리자 메뉴")
            if st.button("매뉴얼 PDF 업로드"):
                st.session_state.show_pdf_upload = True
                st.rerun()
    
    # PDF 업로드 화면 표시
    if st.session_state.get("show_pdf_upload", False):
        upload_pdf()
        return
    
    # 벡터 스토어 확인
    vector_store = get_vector_store()
    if vector_store is None:
        st.warning("매뉴얼 PDF가 업로드되지 않았습니다. 관리자에게 문의하세요.")
        return
    
    # 채팅 UI
    for message in st.session_state.messages:
        avatar = "🧑‍💼" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
    
    # 사용자 입력 처리
    if prompt := st.chat_input("질문을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""
            
            # 챗봇 체인 생성
            chat_chain = get_chat_chain()
            
            # 스트리밍 응답
            for chunk in chat_chain.stream(prompt):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.01)
            
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# 메인 앱 실행
def main():
    # 커스텀 CSS
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f5f7f9;
        }
        .stChatMessage {
            background-color: white;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .stTextInput>div>div>input {
            border-radius: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # OpenAI API 키 확인
    if not OPENAI_API_KEY:
        st.error("OpenAI API 키가 설정되지 않았습니다. Streamlit의 secrets.toml 파일에 OPENAI_API_KEY를 설정해주세요.")
        return
    
    # 로그인 상태에 따른 화면 표시
    if not st.session_state.authenticated:
        login_ui()
    else:
        chatbot_ui()

if __name__ == "__main__":
    main() 
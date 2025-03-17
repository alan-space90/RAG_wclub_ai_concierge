# W Club AI 컨시어지

W Club AI 컨시어지는 고객 문의 게시글에 자동으로 답변하는 RAG(Retrieval Augmented Generation) 기반 챗봇 서비스입니다. 더블유클럽의 매뉴얼 PDF를 기반으로 고객 문의에 친절하고 정확하게 답변합니다.

## 주요 기능

- **로그인 시스템**: 사전에 등록된 이메일/비밀번호 기반 인증 시스템
- **PDF 기반 지식 베이스**: 매뉴얼 PDF를 기반으로 지식 베이스 구축
- **벡터 DB 효율적 관리**: 최초 1회 생성 후 재사용하여 토큰 낭비 방지
- **스트리밍 응답**: 실시간 스트리밍 방식의 응답 제공
- **고급 UI**: 사용자 친화적인 인터페이스

## 설치 및 실행 방법

### 1. 저장소 클론

```bash
git clone https://github.com/username/wclub-ai-concierge.git
cd wclub-ai-concierge
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. API 키 및 암호화 키 설정

`.streamlit/secrets.toml` 파일을 생성하고 필요한 값을 설정합니다. 자세한 내용은 `.streamlit/README.md`를 참조하세요.

### 4. 애플리케이션 실행

```bash
streamlit run app.py
```

## Streamlit Cloud 배포

1. GitHub에 코드를 업로드합니다.
2. [Streamlit Cloud](https://share.streamlit.io/)에 로그인합니다.
3. "New app" 버튼을 클릭하고 GitHub 저장소, 브랜치, 파일을 선택합니다.
4. Secrets 설정에서 필요한 API 키와 암호화 키를 설정합니다.
5. "Deploy" 버튼을 클릭하여 배포합니다.

## 로그인 정보

- 관리자: admin@wclub.com / admin123!
- 게스트: guest@wclub.com / guest123!

## PDF 매뉴얼 업로드

1. 관리자 계정으로 로그인합니다.
2. 사이드바의 "매뉴얼 PDF 업로드" 버튼을 클릭합니다.
3. W Club 매뉴얼 PDF 파일을 업로드합니다.

## 기술 스택

- **Streamlit**: 웹 인터페이스
- **LangChain**: RAG 파이프라인 구축
- **ChromaDB**: 벡터 저장소
- **OpenAI**: 임베딩 및 텍스트 생성
- **PyPDF**: PDF 문서 처리

## 주의사항

- API 키는 반드시 보안을 유지하세요.
- 대용량 PDF 파일은 처리 시간이 오래 걸릴 수 있습니다.
- Streamlit Cloud 무료 티어에는 제한이 있을 수 있으니 주의하세요. 
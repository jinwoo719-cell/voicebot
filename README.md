# 음성 비서 프로그램 (Gemini 버전)

교재 **《진짜 챗GPT API 활용법》 PART 03 '나만의 음성 비서 만들기'** 를
OpenAI(Whisper + GPT) 대신 **Google Gemini API** 로 구현한 실습 프로젝트입니다.

말로 질문하면 → 텍스트로 받아쓰고 → Gemini가 답하고 → 음성으로 읽어 줍니다.

---

## 동작 흐름

```
[사용자 음성]
     │  st.audio_input  (스트림릿 내장 녹음 위젯, 16kHz WAV)
     ▼
[STT]  Gemini 가 오디오를 직접 듣고 받아쓰기          ← 교재는 OpenAI Whisper
     ▼
[대화 기록에 추가]  st.session_state["messages"]
     ▼
[답변 생성]  Gemini generate_content                 ← 교재는 OpenAI GPT
     ▼
[TTS]  gTTS 로 mp3 생성 → HTML autoplay 로 자동 재생   ← 교재와 동일
```

## 교재와 달라진 점

| 구분 | 교재 | 이 프로젝트 | 이유 |
|---|---|---|---|
| STT | OpenAI Whisper API | Gemini 오디오 입력 | API 키 하나로 통일, 임시 파일 불필요 |
| 답변 | OpenAI GPT-4 / GPT-3.5 | Gemini 3.6 Flash / 2.5 Flash | 과제 요구사항 |
| 녹음 | `streamlit-audiorecorder` | `st.audio_input` (내장) | ffmpeg 등 시스템 의존성 제거 → 배포 안정성 |
| 임시 파일 | `input.mp3` / `output.mp3` 저장 후 삭제 | 메모리(BytesIO)에서 처리 | 파일 삭제 실패·동시 접속 충돌 방지 |
| 중복 처리 방지 | `check_reset` 플래그 | 오디오 바이트 해시 비교 | 같은 녹음이 재실행마다 다시 처리되는 문제 해결 |
| 초기화 | 상태 변수만 되돌림 | 녹음 위젯 `key` 회차 부여 | 초기화 후 대화가 되살아나던 버그 수정 |
| 오류 처리 | 없음 | STT·답변·gTTS를 `try/except` | 배포 환경에서 앱이 죽지 않고 원인 안내 |
| API 키 | 사이드바 입력만 | 사이드바 + `st.secrets` | 배포 시 키를 코드에 노출하지 않음 |
| 테스트 | 없음 | `AppTest` 46개 항목 | 회귀 방지 |

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run voicebot.py
```

브라우저가 열리면 사이드바에 Gemini API 키를 붙여 넣고 사용합니다.
키는 [Google AI Studio](https://aistudio.google.com/app/apikey)에서 무료로 발급받습니다.

키를 매번 입력하기 번거로우면:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# secrets.toml 을 열어 실제 키를 입력
```

## 테스트

```bash
python3 test_voicebot.py
```

11개 그룹 46개 항목을 검증합니다 — 앱 기동, 세션 상태, 채팅 렌더링,
녹음→STT→답변→TTS 전체 흐름, 중복 호출 차단, **초기화 버튼 회귀 테스트**,
Gemini 호출 형식(모델명·오디오 바이트·mime type·역할 매핑·시스템 프롬프트 분리),
응답이 `None`일 때의 방어, TTS 언어 판별, gTTS 장애 대응.

**Gemini와 gTTS를 모두 mock으로 대체하므로 API 키도 인터넷도 필요 없습니다.**

## 배포

`배포_가이드.md` 참고. 요약하면:

1. 깃허브에 리포지토리 생성 후 `voicebot.py`, `requirements.txt` 업로드
2. [share.streamlit.io](https://share.streamlit.io) 에서 리포지토리 연결
3. **Advanced settings → Secrets** 에 `GEMINI_API_KEY = "..."` 입력
4. Deploy

## 파일 구성

```
voicebot/
├── voicebot.py                     # 앱 본체
├── test_voicebot.py                # 검증 스크립트
├── requirements.txt                # 파이썬 패키지 (배포 시 필수)
├── .gitignore                      # secrets.toml 커밋 방지
├── .streamlit/
│   └── secrets.toml.example        # 로컬 키 설정 예시
├── README.md
├── 배포_가이드.md
├── 코드리뷰_해설.md                 # 줄단위 코드 설명
├── 예상질문_답변.md                 # 코드리뷰 예상 문답
└── 아키텍처.html                    # 발표용 구조 다이어그램
```

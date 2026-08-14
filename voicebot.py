"""
오늘 뭐 듣지?  —  말로 기분을 말하면 오늘의 한 곡을 골라주는 프로그램

교재 <진짜 챗GPT API 활용법> PART 03 '나만의 음성 비서 만들기'를 바탕으로,
OpenAI(Whisper + GPT) 대신 Google Gemini를 쓰고 목적을 '노래 추천'으로 좁혔습니다.


┌─ 전체 흐름 ────────────────────────────────────────────────────────┐
│                                                                    │
│   사용자가 말한다   "비 오는 날 새벽에 혼자 듣기 좋은 노래"            │
│         │                                       
│
│         ▼   st.audio_input  ......................... 496행         │
│   녹음된 WAV 바이트                                                  │
│         │                                                          │
│         ▼   빈 파일 / 12MB 초과 / 같은 녹음 걸러내기 .... 516행        │
│   통과한 오디오                                                      │
│         │                                                          │
│         ▼   recommend()  ★ 핵심 ..................... 216행         │
│   Gemini 한 번 호출 → JSON 6개 필드                                  │
│     transcript  말한 내용 그대로                                     │
│     title       곡 제목          reason      추천 이유               │
│     artist      가수             background  노래 배경               │
│     one_liner   한 줄 소개  ← 이것만 음성으로 읽음                     │
│         │                                                          │
│         ▼   session_state에 누적 ................... 378행          │
│   대화 기록 (다음 질문에 문맥이 이어짐)                                │
│         │                                                          │
│         ├──▶  render_recommendation()  카드 그리기 ... 319행         │
│         └──▶  TTS()  한 줄 소개만 음성 재생 ........... 275행         │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘


교재와 가장 다른 점 — API를 두 번이 아니라 한 번만 부릅니다

    [교재]    녹음 → Whisper로 받아쓰기 → 그 텍스트를 GPT에 다시 전송 → 답변
              (왕복 2회)

    [본 코드] 녹음 → Gemini에 "오디오 + 지시문"을 한 번에 전송
              → 받아쓴 문장과 추천을 JSON으로 동시에 받음
              (왕복 1회)


실행:  streamlit run voicebot.py
테스트: python3 test_voicebot.py   (API 키·인터넷 없이 115개 항목 검증)
"""

# ══════════════════════════════════════════════════════════════════════
#  1. 패키지 불러오기
# ══════════════════════════════════════════════════════════════════════
import base64                       # mp3를 HTML에 심기 위한 인코딩
import hashlib                      # 같은 녹음을 중복 처리하지 않기 위한 해시
import html                         # 사용자 입력을 HTML에 넣을 때 이스케이프
import json                         # Gemini가 돌려준 JSON 파싱
import re                           # 한글 포함 여부 판별
from datetime import datetime       # 말풍선에 표시할 시각
from io import BytesIO              # gTTS 결과를 메모리에서 다루기 위함
from zoneinfo import ZoneInfo       # 서버가 UTC라서 한국 시간으로 변환

import streamlit as st              # 웹 UI
from google import genai            # Gemini API
from google.genai import types      # 오디오/설정을 넘길 때 쓰는 타입
from gtts import gTTS               # TTS (Google Translate TTS)


# ══════════════════════════════════════════════════════════════════════
#  2. 상수  —  "어떤 형식으로 답하게 할지"를 여기서 정합니다
# ══════════════════════════════════════════════════════════════════════
APP_TITLE = "오늘 뭐 듣지?"
APP_SUBTITLE = "말로 기분을 들려주세요. 오늘의 한 곡을 골라드립니다."

# 배포 서버(Streamlit Cloud)의 시간대는 UTC입니다.
# 그대로 두면 말풍선 시각이 9시간 어긋나므로 한국 시간으로 변환합니다.
KST = ZoneInfo("Asia/Seoul")

# 라디오 버튼에 노출할 Gemini 모델. 두 모델 모두 오디오 입력을 지원합니다.
MODELS = ["gemini-2.5-flash", "gemini-3.6-flash"]

# 오디오 크기 상한.
# Gemini는 인라인 오디오를 포함한 요청 전체를 20MB로 제한합니다.
# base64로 인코딩하면 약 1.33배가 되므로 12MB(약 6분)에서 미리 끊습니다.
MAX_AUDIO_BYTES = 12 * 1024 * 1024

# 데모 앱이므로 한 세션에서 쓸 수 있는 횟수를 제한합니다.
# (공개 배포 + 서버에 저장된 키 조합이라 방문자가 할당량을 소진할 수 있음)
MAX_TURNS_PER_SESSION = 10

# 역할과 답변 형식을 정해 주는 시스템 프롬프트
SYSTEM_PROMPT = """\
You are a warm, knowledgeable music curator for Korean listeners.
The user speaks about their mood, weather, situation, or taste.
Recommend EXACTLY ONE song that fits.

Rules:
- Recommend a real, existing song. Never invent a song or artist.
- Korean and international songs are both fine. Match the user's language and vibe.
- Do not recommend a song you already recommended earlier in this conversation.
- Write every field in Korean, in a warm conversational tone (해요체).
- If the audio has no discernible speech, set transcript to an empty string
  and recommend a song that suits a calm, ordinary day.
"""

# 오디오와 함께 보낼 작업 지시문.
# 받아쓰기와 추천을 한 번에 시켜서 API 왕복을 절반으로 줄입니다.
TASK_PROMPT = """\
Listen to the attached audio and do BOTH tasks in one response:

1. transcript — Write down exactly what the user said, verbatim, in their language.
2. Recommend one song that fits what they said. Fill in:
   - title       : 곡 제목만 (따옴표 없이)
   - artist      : 가수/밴드 이름만
   - one_liner   : 이 곡을 고른 이유를 한 문장으로. 라디오 DJ가 곡을 소개하듯이.
                   30자 내외. 이 문장만 음성으로 읽어 줍니다.
   - reason      : 사용자의 말과 이 곡이 왜 어울리는지 2~3문장.
                   사용자가 말한 내용을 구체적으로 짚어 주세요.
   - background  : 이 곡의 배경 2~3문장. 발매 시기, 만들어진 계기,
                   가사에 담긴 이야기, 화제가 된 지점 등.
                   확실하지 않은 사실은 쓰지 마세요.
"""

# 구조화 출력(response_schema)
#
# 목적을 '노래 추천'으로 좁히니 필요한 항목이 정해졌고,
# 그러자 아래처럼 스키마로 못 박아 카드 UI를 그릴 수 있음.
RECOMMENDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "transcript": {"type": "string"},
        "title": {"type": "string"},
        "artist": {"type": "string"},
        "one_liner": {"type": "string"},
        "reason": {"type": "string"},
        "background": {"type": "string"},
    },
    "required": ["transcript", "title", "artist", "one_liner", "reason", "background"],
    "propertyOrdering": [
        "transcript", "title", "artist", "one_liner", "reason", "background",
    ],
}


# ══════════════════════════════════════════════════════════════════════
#  3. 기능 구현 함수
#
#     load_default_apikey()   배포 환경의 Secrets에서 API 키 읽기
#     normalize_mime()        브라우저마다 다른 오디오 형식 이름 맞추기
#     get_client()            Gemini 클라이언트 (타임아웃·재시도 설정)
#     recommend()             ★ 받아쓰기 + 추천을 한 번에            <- 핵심
#     TTS()                   한 줄 소개를 mp3로 만들어 자동 재생
#     render_user_bubble()    말한 내용을 파란 말풍선으로
#     render_recommendation() 추천 결과를 카드로
# ══════════════════════════════════════════════════════════════════════
def load_default_apikey() -> str:
    """배포 환경(Streamlit Cloud)의 Secrets에 키가 있으면 그것을 씁니다.

    로컬에는 보통 .streamlit/secrets.toml이 없고, 그 상태로 st.secrets에 접근하면
    StreamlitSecretNotFoundError가 나서 앱이 죽습니다. 그래서 감싸 줍니다.
    """
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


def normalize_mime(mime_type: str) -> str:
    """브라우저가 알려주는 mime type을 Gemini가 아는 이름으로 맞춰 줍니다.

    같은 형식이라도 브라우저·OS마다 다르게 부릅니다.
    (예: mp3를 audio/mpeg, wav를 audio/x-wav 로 보내는 경우)
    """
    aliases = {
        "audio/mpeg": "audio/mp3",
        "audio/mpeg3": "audio/mp3",
        "audio/x-mpeg-3": "audio/mp3",
        "audio/x-wav": "audio/wav",
        "audio/wave": "audio/wav",
        "audio/vnd.wave": "audio/wav",
        "audio/x-flac": "audio/flac",
        "audio/x-aac": "audio/aac",
    }
    mime_type = (mime_type or "audio/wav").split(";")[0].strip().lower()
    return aliases.get(mime_type, mime_type)


def get_client(apikey: str) -> genai.Client:
    """API 키로 Gemini 클라이언트를 만듭니다. (교재의 openai.OpenAI()에 해당)

    타임아웃과 재시도 횟수를 지정하는 이유:
    기본값으로 두면 요청이 막혔을 때 SDK가 오래 재시도해서, 화면에는 스피너만
    계속 돌고 사용자는 이유를 알 수 없습니다. 60초 안에 실패시키고 원인을
    화면에 보여 주는 편이 낫습니다.
    """
    return genai.Client(
        api_key=apikey,
        http_options=types.HttpOptions(
            timeout=60_000,  # 60초 (밀리초 단위)
            retry_options=types.HttpRetryOptions(attempts=2),
        ),
    )


# ─────────────────────────────────────────────────────────────────────
#  ★ 이 프로젝트의 핵심 함수
#
#  교재는 이렇게 두 번 부릅니다
#      STT(audio)      → Whisper가 받아쓴 텍스트
#      ask_gpt(텍스트)  → GPT가 만든 답변
#
#  Gemini는 오디오와 지시문을 함께 받을 수 있어서 한 번이면 됩니다
#      recommend(audio) → 받아쓴 문장 + 추천을 JSON으로 한꺼번에
# ─────────────────────────────────────────────────────────────────────
def recommend(audio_bytes: bytes, mime_type: str, history: list,
              apikey: str, model: str) -> dict:
    """오디오를 넘겨 '받아쓰기 + 노래 추천'을 한 번에 받아 옵니다."""
    client = get_client(apikey)

    # 이전 대화를 먼저 넣어야 "아까랑 다른 걸로" 같은 후속 요청이 통합니다.
    contents = [
        types.Content(role=m["role"], parts=[types.Part(text=m["content"])])
        for m in history
    ]
    # 지시문과 오디오를 함께 넣어 보냄. "받아쓰고, 그 내용에 맞는 곡을 고름."
    # STT(오디오) 로 글자를 얻고, 그 글자를 ask_gpt()에 다시 보냄
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part(text=TASK_PROMPT),
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            ],
        )
    )

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=RECOMMENDATION_SCHEMA,
    )

    # Gemini 2.5 계열은 '생각하기'가 기본으로 켜져 있습니다.
    # 노래 한 곡 고르는 작업에는 과한 기능이라, 껐더니 응답이 눈에 띄게 빨라졌습니다.
    # (Gemini 3 계열은 설정 이름이 달라서 손대지 않습니다)
    if model.startswith("gemini-2.5"):
        config.thinking_config = types.ThinkingConfig(thinking_budget=0)

    response = client.models.generate_content(
        model=model, contents=contents, config=config
    )

    # 스키마를 강제했어도 안전 필터나 토큰 초과로 응답이 깨질 수 있습니다.
    raw = (response.text or "").strip()
    if not raw:
        raise ValueError("Gemini가 빈 응답을 돌려주었습니다. 다시 시도해 주세요.")

    data = json.loads(raw)

    # json.loads는 문자열이나 배열도 통과시킵니다.
    # 여기서 걸러내지 않으면 화면을 그릴 때 AttributeError로 앱이 죽습니다.
    if not isinstance(data, dict):
        raise ValueError("Gemini 응답이 JSON 객체가 아닙니다.")

    # 스키마가 필수라고 해도 빈 문자열은 통과합니다.
    # 빈 카드가 그려지고 대화 기록까지 오염되므로 여기서 막습니다.
    missing = [k for k in ("title", "artist", "one_liner")
               if not str(data.get(k, "")).strip()]
    if missing:
        raise ValueError(f"응답에 빠진 항목이 있습니다: {', '.join(missing)}")

    return data


def TTS(text: str) -> None:
    """텍스트 -> 음성. 교재와 동일하게 gTTS를 쓰되, 디스크 대신 메모리에서 처리합니다.

    추천 이유와 배경까지 다 읽으면 생성이 오래 걸려서,
    한 줄 소개(one_liner)만 읽어 줍니다.
    """
    lang = "ko" if re.search(r"[가-힣]", text) else "en"

    # gTTS는 구글 번역 서버를 호출하므로 실패할 수 있습니다.
    # 소리가 안 나는 것보다 추천 내용이 사라지는 게 더 큰 문제라 여기서 막습니다.
    try:
        buffer = BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(buffer)
        audio_bytes = buffer.getvalue()
    except Exception as e:
        st.warning(f"음성 변환에 실패했습니다(추천 내용은 위에 있습니다): {e}")
        return

    # 음원 자동 재생 (교재와 동일한 HTML autoplay 방식)
    b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(
        f"""
        <audio autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """,
        unsafe_allow_html=True,
    )

    # 브라우저가 자동 재생을 막았을 때를 대비한 수동 재생 플레이어
    st.audio(audio_bytes, format="audio/mp3")


def render_user_bubble(text: str, time: str) -> None:
    """사용자가 말한 내용을 파란 말풍선으로 그립니다. (교재의 채팅 UI)"""
    st.markdown(
        f'<div style="display:flex;align-items:center;margin-bottom:0.5rem;">'
        f'<div style="background-color:#007AFF;color:white;border-radius:14px;'
        f'padding:8px 14px;margin-right:8px;max-width:80%;">{html.escape(text)}</div>'
        f'<div style="font-size:0.8rem;color:gray;">{html.escape(time)}</div></div>',
        unsafe_allow_html=True,
    )


def render_recommendation(rec: dict, time: str) -> None:
    """추천 결과를 카드 형태로 그립니다."""
    title = html.escape(rec.get("title", ""))
    artist = html.escape(rec.get("artist", ""))
    one_liner = html.escape(rec.get("one_liner", ""))
    reason = html.escape(rec.get("reason", ""))
    background = html.escape(rec.get("background", ""))

    # ⚠️ 카드 배경을 밝은 색으로 고정했으므로 글자색도 반드시 함께 고정해야 합니다.
    # 색을 지정하지 않으면 스트림릿 테마의 글자색을 물려받는데,
    # 다크 모드에서는 그 색이 흰색이라 흰 배경 위에서 글자가 보이지 않습니다.
    # (모바일 다크 모드에서 제목과 본문이 통째로 안 보이는 버그를 겪었습니다)
    st.markdown(
        f"""
        <div style="border:1px solid #e3e6ea;border-radius:14px;padding:18px 20px;
                    background:#fafbfc;color:#1a1d21;margin-bottom:1.2rem;">
          <div style="font-size:1.25rem;font-weight:700;line-height:1.3;color:#1a1d21;">
            🎵 {title}
          </div>
          <div style="color:#6b7280;font-size:0.95rem;margin-top:2px;">{artist}</div>
          <div style="margin-top:12px;padding:10px 14px;background:#eef4ff;
                      color:#1a1d21;border-radius:10px;font-size:0.95rem;">
            {one_liner}
          </div>
          <div style="margin-top:16px;">
            <div style="font-size:0.75rem;letter-spacing:0.04em;color:#6b7280;
                        font-weight:700;">추천 이유</div>
            <div style="margin-top:4px;line-height:1.65;color:#1a1d21;">{reason}</div>
          </div>
          <div style="margin-top:14px;">
            <div style="font-size:0.75rem;letter-spacing:0.04em;color:#6b7280;
                        font-weight:700;">노래 배경</div>
            <div style="margin-top:4px;line-height:1.65;color:#1a1d21;">{background}</div>
          </div>
          <div style="text-align:right;font-size:0.8rem;color:gray;margin-top:10px;">
            {html.escape(time)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
#  4. 메인 함수  —  화면 구성과 처리 순서
#
#     스트림릿은 버튼 하나만 눌러도 이 함수를 처음부터 끝까지 다시 실행합니다.
#     그래서 유지돼야 하는 값은 전부 st.session_state에 넣습니다.
#
#     [세션 상태]  chat  messages  GEMINI_API  last_audio_id
#                  audio_round  last_error
#     [사이드바]    API 키 상태 · 모델 선택 · 초기화
#     [왼쪽 col1]   녹음 위젯 → 가드 → recommend() 호출 → 상태 저장
#     [오른쪽 col2] 카드 렌더링 → 한 줄 소개 음성 재생
# ══════════════════════════════════════════════════════════════════════
def main():
    # --- 기본 설정 ---
    st.set_page_config(page_title=APP_TITLE, page_icon="🎧", layout="wide")

    # --- 세션 상태 초기화 ---
    # 화면에 그릴 기록. 한 턴 = {"time": 시각, "said": 말한 내용, "rec": 추천 결과}
    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    # Gemini에 넘길 대화 기록: {"role": "user"/"model", "content": ...}
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # API 키
    if "GEMINI_API" not in st.session_state:
        st.session_state["GEMINI_API"] = ""

    # 같은 녹음을 다시 처리하지 않기 위한 직전 녹음의 지문(해시)
    if "last_audio_id" not in st.session_state:
        st.session_state["last_audio_id"] = None

    # 녹음 위젯 key에 붙일 회차 번호.
    # 초기화할 때 이 번호를 올리면 위젯이 새것으로 교체되어 녹음이 비워집니다.
    if "audio_round" not in st.session_state:
        st.session_state["audio_round"] = 0

    # 직전 실패 사유. 재실행돼도 화면에 남기기 위해 세션에 보관합니다.
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = None

    # 이번 실행에서 새로 나온 추천 (있으면 마지막에 음성으로 읽어 줍니다)
    new_one_liner = None

    # --- 제목 ---
    st.title(f"🎧 {APP_TITLE}")
    st.caption(APP_SUBTITLE)
    st.markdown("---")

    # --- 기본 설명 ---
    with st.expander("이 프로그램에 관하여", expanded=False):
        st.write(
            """
            - UI는 **스트림릿(Streamlit)** 으로 만들었습니다.
            - 음성 인식(STT)과 노래 추천을 **Google Gemini** 가 한 번에 처리합니다.
              (교재는 Whisper + GPT를 따로 불러 API를 2번 호출합니다)
            - 답변 형식은 **구조화 출력(response_schema)** 으로 고정했습니다.
            - 한 줄 소개는 구글의 **Google Translate TTS(gTTS)** 로 읽어 줍니다.
            """
        )

    # --- 사이드바 ---
    with st.sidebar:
        # API 키 확보
        #
        # ⚠️ 보안상 중요한 부분입니다.
        # Secrets에 있는 키를 text_input의 value로 넣으면 그 값이 브라우저까지
        # 전송됩니다. type="password"는 눈으로만 가릴 뿐이라, 눈 아이콘을 누르거나
        # 개발자 도구를 열면 그대로 보입니다. 배포된 앱은 누구나 접속하므로
        # 그건 키를 공개하는 것과 같습니다.
        #
        # 그래서 Secrets에 키가 있으면 입력창을 아예 만들지 않고,
        # 서버 쪽 session_state에만 담아 씁니다. 키는 브라우저로 나가지 않습니다.
        secret_key = load_default_apikey()

        if secret_key:
            st.session_state["GEMINI_API"] = secret_key
            st.success("API 키가 설정되어 있습니다.", icon="🔑")
            st.caption("서버(Secrets)에 저장된 키를 사용합니다.")
        else:
            st.session_state["GEMINI_API"] = st.text_input(
                label="GEMINI API 키",
                placeholder="Enter your API Key",
                value=st.session_state["GEMINI_API"],
                type="password",
            )
            st.markdown(
                "[API 키 발급받기](https://aistudio.google.com/app/apikey)",
                help="Google AI Studio에서 무료로 발급받을 수 있습니다.",
            )
        st.markdown("---")

        # 모델 선택
        model = st.radio(label="Gemini 모델", options=MODELS)
        st.markdown("---")

        # ★ 발표 포인트 ─ 초기화 버튼에서 만난 버그
        #
        # 처음에는 chat / messages / last_audio_id 만 비웠습니다. 그랬더니
        # 초기화를 눌러도 지운 대화가 곧바로 되살아났습니다.
        #
        #   초기화 → 상태 비움 → st.rerun()
        #     → 재실행: 녹음 위젯에는 아까 녹음이 그대로 남아 있음
        #     → 해시가 None과 다르니 "새 녹음이네!" → 처음부터 다시 처리
        #
        # 아래 audio_round 를 올리면 녹음 위젯의 key가 바뀝니다.
        # 스트림릿은 key가 다르면 '다른 위젯'으로 보고 빈 상태로 새로 만듭니다.
        if st.button(label="초기화"):
            st.session_state["chat"] = []
            st.session_state["messages"] = []
            st.session_state["last_audio_id"] = None
            st.session_state["last_error"] = None
            # ⚠️ 회차를 올려 녹음 위젯을 새것으로 갈아 끼웁니다.
            # 이게 없으면 위젯에 남아 있던 녹음이 그대로 반환되고,
            # last_audio_id를 None으로 비운 탓에 '새 녹음'으로 판정되어
            # 방금 지운 질문이 즉시 다시 처리됩니다. (실제로 겪은 버그)
            st.session_state["audio_round"] += 1
            st.rerun()

        st.caption(
            "예) 비 오는 날 듣기 좋은 노래 / 운동할 때 들을 신나는 곡 / "
            "새벽에 혼자 듣고 싶은 노래"
        )

    # --- 기능 구현 공간 ---
    col1, col2 = st.columns([1, 1.4])

    with col1:
        # 왼쪽 영역: 말하기
        st.subheader("말하기")

        # 스트림릿 내장 녹음 위젯. 16kHz WAV 바이트를 돌려줍니다.
        # key에 회차 번호를 넣어, 초기화할 때 위젯 자체가 비워지도록 합니다.
        audio = st.audio_input(
            "클릭하여 녹음하기",
            key=f"audio_{st.session_state['audio_round']}",
        )

        # 마이크를 쓸 수 없는 환경(장치 없음, 회의실 PC 등)을 위한 대체 입력
        if audio is None:
            audio = st.file_uploader(
                "마이크가 안 되면 음성 파일을 올리세요 (wav, mp3)",
                type=["wav", "mp3", "ogg", "flac", "aac"],
                key=f"upload_{st.session_state['audio_round']}",
            )

        if audio is not None:
            audio_bytes = audio.getvalue()
            # 녹음 내용의 해시를 지문으로 삼아, 새 녹음일 때만 처리합니다.
            # (보안이 아니라 '같은 데이터인지' 비교용이라 md5로 충분합니다)
            audio_id = hashlib.md5(audio_bytes, usedforsecurity=False).hexdigest()
            is_new_audio = audio_id != st.session_state["last_audio_id"]

            if not audio_bytes:
                st.error("빈 오디오입니다. 다시 녹음해 주세요.")

            elif len(audio_bytes) > MAX_AUDIO_BYTES:
                # Gemini는 인라인 오디오 요청을 20MB로 제한합니다.
                # 넘겨보고 실패하는 대신 여기서 미리 안내합니다.
                st.error(
                    f"녹음이 너무 깁니다 ({len(audio_bytes) / 1048576:.1f}MB). "
                    f"{MAX_AUDIO_BYTES // 1048576}MB(약 6분) 이하로 해 주세요."
                )

            elif not is_new_audio:
                # 같은 녹음이 다시 들어온 경우. 조용히 넘어가면 사용자는
                # 앱이 멈춘 줄 알기 때문에, 왜 아무 일도 없는지 알려 줍니다.
                if st.session_state["chat"]:
                    st.caption(
                        "방금 처리한 것과 같은 녹음입니다. "
                        "새로 녹음하거나 사이드바에서 초기화해 주세요."
                    )

            elif not st.session_state["GEMINI_API"]:
                st.warning("사이드바에 Gemini API 키를 입력해 주세요.")

            elif len(st.session_state["chat"]) >= MAX_TURNS_PER_SESSION:
                st.warning(
                    f"데모 앱이라 한 번에 {MAX_TURNS_PER_SESSION}곡까지만 추천합니다. "
                    "사이드바에서 초기화해 주세요."
                )

            else:
                # API 호출 '전에' 기록합니다. 실패해도 무한 재시도하지 않도록.
                st.session_state["last_audio_id"] = audio_id

                with st.spinner("듣고 있어요... 오늘의 한 곡을 고르는 중입니다"):
                    try:
                        rec = recommend(
                            audio_bytes,
                            normalize_mime(audio.type),
                            st.session_state["messages"],
                            st.session_state["GEMINI_API"],
                            model,
                        )
                        st.session_state["last_error"] = None
                    except Exception as e:
                        rec = None
                        # 에러를 세션에 남깁니다. 여기서 st.error만 부르면
                        # 다음 재실행 때 메시지가 사라져 원인을 볼 수 없습니다.
                        st.session_state["last_error"] = str(e)

                if rec:
                    now = datetime.now(KST).strftime("%H:%M")
                    said = (rec.get("transcript") or "").strip()

                    # 화면에 그릴 기록. 한 번의 대화를 한 덩어리로 묶어 둡니다.
                    # (말한 내용과 추천이 항상 짝을 이뤄 순서가 어긋나지 않습니다)
                    st.session_state["chat"].append(
                        {"time": now, "said": said, "rec": rec}
                    )

                    # 다음 요청에 문맥이 이어지도록 대화 기록에도 저장.
                    # 오디오는 넣지 않고 텍스트만 쌓아서 토큰을 아낍니다.
                    st.session_state["messages"].append(
                        {"role": "user", "content": said or "(무음)"}
                    )
                    st.session_state["messages"].append(
                        {
                            "role": "model",
                            "content": f"{rec.get('title')} - {rec.get('artist')} 추천함",
                        }
                    )

                    new_one_liner = rec.get("one_liner")

        # 실패했다면 원인을 보여 주고, 같은 녹음으로 다시 시도할 길을 열어 둡니다.
        # (해시 가드 때문에 그냥 두면 같은 녹음으로는 영영 재시도할 수 없습니다)
        if st.session_state["last_error"]:
            st.error(f"추천에 실패했습니다: {st.session_state['last_error']}")
            if st.button("같은 녹음으로 다시 시도"):
                st.session_state["last_audio_id"] = None
                st.session_state["last_error"] = None
                st.rerun()

    with col2:
        # 오른쪽 영역: 추천 결과
        st.subheader("오늘의 추천")

        if not st.session_state["chat"]:
            st.info("왼쪽에서 녹음 버튼을 누르고, 지금 기분이나 상황을 말해 보세요.")

        # 최근 추천이 맨 위로 오도록 뒤집어서 그립니다.
        for entry in reversed(st.session_state["chat"]):
            if entry["said"]:
                render_user_bubble(entry["said"], entry["time"])
            render_recommendation(entry["rec"], entry["time"])

        # 한 줄 소개만 음성으로 재생 (이번 실행에서 새로 나온 추천이 있을 때만)
        if new_one_liner:
            TTS(new_one_liner)


if __name__ == "__main__":
    main()

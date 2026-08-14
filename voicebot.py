"""
오늘 뭐 듣지?  —  공부용 (모든 줄에 주석)

배포용 voicebot.py 와 동작은 완전히 같습니다.
이해한 줄의 주석은 지워 가면서 보세요. 다 지우면 배포용과 같아집니다.
"""

# ══════════════════════════════════════════════════════════════════════
#  1. 패키지 불러오기
# ══════════════════════════════════════════════════════════════════════
import base64                    # mp3 바이트 → 문자열. HTML 안에 소리를 직접 심을 때 씀
import hashlib                   # 데이터의 '지문'을 만드는 도구. 같은 녹음인지 비교용
import html                      # <, >, " 같은 글자를 안전한 형태로 바꿔줌 (해킹 방지)
import json                      # Gemini가 준 JSON 글자를 파이썬 딕셔너리로 바꿈
import re                        # 정규식. 여기서는 '한글이 들어있나' 확인에만 씀
from datetime import datetime    # 현재 시각을 구함
from io import BytesIO           # 파일처럼 쓰는 메모리 공간. 디스크에 안 쓰고 처리
from zoneinfo import ZoneInfo    # 시간대 정보. 서버가 UTC라서 한국 시간으로 바꿔야 함

import streamlit as st           # 웹 화면을 만드는 도구. st. 으로 줄여 씀
from google import genai         # Gemini를 부르는 공식 도구
from google.genai import types   # Gemini에 넘길 데이터의 '틀'들 (Content, Part 등)
from gtts import gTTS            # 글자를 사람 목소리 mp3로 바꿔주는 도구


# ══════════════════════════════════════════════════════════════════════
#  2. 상수  —  프로그램 전체에서 쓰는 고정값. 대문자로 쓰는 게 관례
# ══════════════════════════════════════════════════════════════════════
APP_TITLE = "오늘 뭐 듣지?"                                    # 앱 제목. 여러 곳에서 재사용
APP_SUBTITLE = "말로 기분을 들려주세요. 오늘의 한 곡을 골라드립니다."  # 제목 밑 설명 문구

KST = ZoneInfo("Asia/Seoul")     # 한국 시간대 객체. 서버는 UTC라 이걸로 변환해야 9시간 안 밀림

MODELS = ["gemini-2.5-flash", "gemini-3.6-flash"]   # 사이드바 라디오에 띄울 모델 목록

MAX_AUDIO_BYTES = 12 * 1024 * 1024   # 12MB. Gemini 요청 한도가 20MB인데 base64로 1.33배 커짐

MAX_TURNS_PER_SESSION = 10       # 한 세션에서 추천받을 수 있는 최대 횟수 (할당량 보호)

# 모델에게 '너는 누구이고 어떻게 답해야 하는지' 알려주는 글.
# 매 요청마다 대화와 별도로 붙여서 보냅니다.
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

# 오디오와 함께 보낼 '이번에 할 일' 지시문.
# 받아쓰기(1번)와 추천(2번)을 한 번에 시켜서 API 왕복을 절반으로 줄입니다.
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

# ★ 답변의 '모양'을 미리 정해두는 설계도.
#   이걸 넘기면 모델이 이 형태를 벗어난 답을 만들 수 없습니다.
RECOMMENDATION_SCHEMA = {
    "type": "object",                       # 최상위는 딕셔너리 하나 (목록 아님)
    "properties": {                         # 그 안에 들어갈 항목들
        "transcript": {"type": "string"},   # 사용자가 말한 내용 그대로
        "title": {"type": "string"},        # 곡 제목
        "artist": {"type": "string"},       # 가수 이름
        "one_liner": {"type": "string"},    # 한 줄 소개 (이것만 음성으로 읽음)
        "reason": {"type": "string"},       # 추천 이유
        "background": {"type": "string"},   # 노래 배경
    },
    "required": [                           # 이 항목들은 반드시 있어야 함
        "transcript", "title", "artist", "one_liner", "reason", "background",
    ],
    "propertyOrdering": [                   # 모델이 이 순서대로 채우게 유도 (품질에 도움)
        "transcript", "title", "artist", "one_liner", "reason", "background",
    ],
}


# ══════════════════════════════════════════════════════════════════════
#  3. 기능 구현 함수
# ══════════════════════════════════════════════════════════════════════
def load_default_apikey() -> str:
    """배포 환경(Streamlit Cloud)의 Secrets에서 API 키를 읽어옵니다."""
    try:                                          # 실패할 수 있는 코드를 감싸는 문법
        return st.secrets.get("GEMINI_API_KEY", "")   # 있으면 키를, 없으면 빈 문자열을 반환
    except Exception:                             # 어떤 오류든 잡음
        return ""                                 # 로컬엔 secrets 파일이 없으니 빈 값으로 넘김
        # ↑ try 없이 st.secrets에 접근하면 파일이 없을 때 앱이 통째로 죽습니다


def normalize_mime(mime_type: str) -> str:
    """브라우저가 알려준 오디오 형식 이름을 Gemini가 아는 이름으로 맞춰줍니다."""
    aliases = {                              # 왼쪽(브라우저가 주는 이름) → 오른쪽(Gemini가 아는 이름)
        "audio/mpeg": "audio/mp3",           # 크롬은 mp3를 audio/mpeg라고 부름
        "audio/mpeg3": "audio/mp3",
        "audio/x-mpeg-3": "audio/mp3",
        "audio/x-wav": "audio/wav",          # 일부 브라우저는 wav를 x-wav라고 부름
        "audio/wave": "audio/wav",
        "audio/vnd.wave": "audio/wav",
        "audio/x-flac": "audio/flac",
        "audio/x-aac": "audio/aac",
    }
    mime_type = (mime_type or "audio/wav").split(";")[0].strip().lower()
    # ↑ 한 줄에 4가지를 함: None이면 wav로 / ";" 뒤 옵션 제거 / 공백 제거 / 소문자로
    return aliases.get(mime_type, mime_type)  # 표에 있으면 바꾸고, 없으면 원래 값 그대로


def get_client(apikey: str) -> genai.Client:
    """Gemini에 요청을 보낼 '연결 담당자'를 만듭니다."""
    return genai.Client(                          # 클라이언트 객체를 만들어 돌려줌
        api_key=apikey,                           # 누구 계정으로 부를지
        http_options=types.HttpOptions(           # 통신 관련 설정 묶음
            timeout=60_000,                       # 60초 (밀리초 단위). 넘으면 포기하고 오류
            retry_options=types.HttpRetryOptions(attempts=2),  # 실패 시 최대 2번만 시도
        ),
        # ↑ 이 설정이 없으면 응답이 안 올 때 화면에 스피너만 계속 돌고 이유를 알 수 없습니다
    )


# ─────────────────────────────────────────────────────────────────────
#  ★ 이 프로젝트의 핵심 함수
#     교재: STT(오디오) → 텍스트,  ask_gpt(텍스트) → 답변   (API 2번)
#     여기: recommend(오디오) → 받아쓴 문장 + 추천 한꺼번에  (API 1번)
# ─────────────────────────────────────────────────────────────────────
def recommend(audio_bytes: bytes, mime_type: str, history: list,
              apikey: str, model: str) -> dict:
    # audio_bytes: 녹음된 소리 데이터 / mime_type: 그 소리의 형식
    # history: 지금까지의 대화 / apikey: 인증키 / model: 어떤 모델을 쓸지
    """오디오를 넘겨 '받아쓰기 + 노래 추천'을 한 번에 받아옵니다."""
    client = get_client(apikey)               # 위에서 만든 함수로 연결 담당자 준비

    contents = [                              # Gemini에 보낼 '대화 목록'을 만듦
        types.Content(role=m["role"], parts=[types.Part(text=m["content"])])
        # ↑ 우리 형식 {"role":..., "content":...} 을 Gemini 형식으로 변환
        for m in history                      # 이전 대화를 하나씩 꺼내서
    ]
    # ↑ 이전 대화를 먼저 넣어야 "아까랑 다른 걸로" 같은 후속 요청이 통합니다

    contents.append(                          # 목록 맨 뒤에 '이번 차례'를 붙임
        types.Content(
            role="user",                      # 이번 발언자는 사용자
            parts=[                           # 한 발언 안에 여러 조각을 넣을 수 있음
                types.Part(text=TASK_PROMPT),                              # 조각1: 지시문
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),  # 조각2: 오디오
                # ↑ 지시문과 오디오를 같이 보내는 것이 이 프로젝트의 핵심입니다
            ],
        )
    )

    config = types.GenerateContentConfig(     # 요청에 붙일 설정 묶음
        system_instruction=SYSTEM_PROMPT,     # 역할 지시 (대화 목록이 아니라 설정으로 전달)
        response_mime_type="application/json",  # "JSON으로 답해라"
        response_schema=RECOMMENDATION_SCHEMA,  # "이 모양으로 답해라"
    )

    if model.startswith("gemini-2.5"):        # 모델 이름이 gemini-2.5로 시작하면
        config.thinking_config = types.ThinkingConfig(thinking_budget=0)
        # ↑ 2.5 계열은 '생각하기'가 기본 ON. 노래 고르기엔 과해서 끔 → 훨씬 빨라짐
        #   (3 계열은 설정 이름이 달라서 건드리지 않음)

    response = client.models.generate_content(   # ★ 실제로 Gemini를 부르는 한 줄
        model=model, contents=contents, config=config
    )

    raw = (response.text or "").strip()       # 답변 글자를 꺼냄. None이면 빈 문자열로, 앞뒤 공백 제거
    # ↑ or "" 가 없으면 None.strip() 에서 오류가 납니다

    if not raw:                               # 빈 문자열이면 (= 답이 안 왔으면)
        raise ValueError("Gemini가 빈 응답을 돌려주었습니다. 다시 시도해 주세요.")
        # ↑ raise = 오류를 일부러 발생시킴. 부른 쪽에서 잡아서 화면에 보여줍니다

    data = json.loads(raw)                    # JSON 글자 → 파이썬 딕셔너리로 변환

    if not isinstance(data, dict):            # 딕셔너리가 맞는지 확인
        raise ValueError("Gemini 응답이 JSON 객체가 아닙니다.")
        # ↑ json.loads는 "문자열"이나 [배열]도 통과시킵니다.
        #   그대로 두면 나중에 .get() 을 부를 때 앱이 죽습니다

    missing = [k for k in ("title", "artist", "one_liner")   # 이 3개 항목 중에서
               if not str(data.get(k, "")).strip()]          # 값이 비어 있는 것만 모음
    if missing:                               # 하나라도 비었으면
        raise ValueError(f"응답에 빠진 항목이 있습니다: {', '.join(missing)}")
        # ↑ 스키마의 required는 '키가 있는지'만 보고 '값이 찼는지'는 안 봅니다.
        #   빈 값을 통과시키면 빈 카드가 그려지고 대화 기록까지 오염됩니다

    return data                               # 검사를 다 통과한 딕셔너리를 돌려줌


def TTS(text: str) -> None:
    """글자를 음성으로 바꿔 자동 재생합니다. (반환값 없음 → None)"""
    lang = "ko" if re.search(r"[가-힣]", text) else "en"
    # ↑ 한글이 한 글자라도 있으면 "ko", 아니면 "en". 영어를 한국어 발음으로 읽는 걸 막음

    try:                                      # gTTS는 구글 서버를 부르므로 실패할 수 있음
        buffer = BytesIO()                    # 메모리 위의 빈 파일 공간을 만듦
        gTTS(text=text, lang=lang).write_to_fp(buffer)   # 음성을 만들어 그 공간에 씀
        audio_bytes = buffer.getvalue()       # 공간에 담긴 mp3 데이터를 꺼냄
    except Exception as e:                    # 실패하면 (as e = 오류 내용을 e에 담음)
        st.warning(f"음성 변환에 실패했습니다(추천 내용은 위에 있습니다): {e}")
        return                                # 함수를 여기서 끝냄. 앱은 계속 살아 있음
        # ↑ 소리가 안 나는 것보다 추천 글이 사라지는 게 더 큰 문제라 여기서 막습니다

    b64 = base64.b64encode(audio_bytes).decode()   # mp3 데이터 → 글자로 변환 (HTML에 넣으려고)
    st.markdown(                              # HTML을 화면에 직접 넣음
        f"""
        <audio autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """,
        unsafe_allow_html=True,               # "HTML을 글자가 아니라 태그로 해석해라"
        # ↑ 스트림릿엔 '자동 재생' 기능이 없어서 HTML 태그를 직접 심는 우회법입니다
    )

    st.audio(audio_bytes, format="audio/mp3")  # 브라우저가 자동재생을 막을 때를 위한 수동 재생기


def render_user_bubble(text: str, time: str) -> None:
    """사용자가 말한 내용을 파란 말풍선으로 그립니다."""
    st.markdown(
        f'<div style="display:flex;align-items:center;margin-bottom:0.5rem;">'   # 가로 배치
        f'<div style="background-color:#007AFF;color:white;border-radius:14px;'  # 파란 배경, 둥근 모서리
        f'padding:8px 14px;margin-right:8px;max-width:80%;">{html.escape(text)}</div>'
        # ↑ html.escape = 사용자 말에 <script> 같은 게 있어도 글자로만 보이게 함 (보안)
        f'<div style="font-size:0.8rem;color:gray;">{html.escape(time)}</div></div>',  # 옆에 시각
        unsafe_allow_html=True,
    )


def render_recommendation(rec: dict, time: str) -> None:
    """추천 결과를 카드 형태로 그립니다. rec = recommend()가 돌려준 딕셔너리"""
    title = html.escape(rec.get("title", ""))          # 곡 제목 꺼내기 (없으면 빈 문자열)
    artist = html.escape(rec.get("artist", ""))        # 가수
    one_liner = html.escape(rec.get("one_liner", ""))  # 한 줄 소개
    reason = html.escape(rec.get("reason", ""))        # 추천 이유
    background = html.escape(rec.get("background", ""))  # 노래 배경
    # ↑ 5개 모두 escape 처리. 하나라도 빠뜨리면 그 자리로 공격이 들어올 수 있습니다

    st.markdown(
        f"""
        <div style="border:1px solid #e3e6ea;border-radius:14px;padding:18px 20px;
                    background:#fafbfc;color:#1a1d21;margin-bottom:1.2rem;">
                    <!-- ↑ 밝은 배경(#fafbfc)을 정했으면 글자색(#1a1d21)도 반드시 함께 정해야 함.
                           안 그러면 다크 모드에서 흰 배경에 흰 글씨가 됩니다 (실제로 겪은 버그) -->
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
#  4. 메인 함수  —  실제로 화면이 만들어지는 곳
#
#     ※ 스트림릿의 가장 중요한 성질:
#        버튼을 누르든 입력을 하든, 이 함수가 처음부터 끝까지 다시 실행됩니다.
#        그래서 보통 변수는 매번 초기화됩니다.
#        살아남아야 하는 값은 st.session_state 라는 특별한 저장소에 넣습니다.
# ══════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🎧", layout="wide")
    # ↑ 브라우저 탭 제목 / 탭 아이콘 / 화면을 넓게 쓸지. 반드시 st 명령 중 맨 처음이어야 함

    # ---------- 세션 상태 초기화 ----------
    # 패턴이 모두 같습니다: "이 값이 아직 없으면 → 처음 값을 넣어라"
    # 이미 있으면 건드리지 않으므로, 재실행돼도 값이 유지됩니다.

    if "chat" not in st.session_state:        # 화면에 그릴 기록이 아직 없으면
        st.session_state["chat"] = []         # 빈 목록으로 시작
        # 한 칸 = {"time": 시각, "said": 말한 내용, "rec": 추천 결과}

    if "messages" not in st.session_state:    # Gemini에 넘길 대화 기록
        st.session_state["messages"] = []     # 한 칸 = {"role": user/model, "content": 글}

    if "GEMINI_API" not in st.session_state:  # API 키
        st.session_state["GEMINI_API"] = ""

    if "last_audio_id" not in st.session_state:   # 직전에 처리한 녹음의 지문(해시)
        st.session_state["last_audio_id"] = None  # 같은 녹음을 두 번 처리하지 않으려고

    if "audio_round" not in st.session_state:     # 녹음 위젯 이름표에 붙일 회차 번호
        st.session_state["audio_round"] = 0       # 초기화할 때 이 숫자를 올림

    if "last_error" not in st.session_state:      # 직전 실패 사유
        st.session_state["last_error"] = None     # 재실행돼도 화면에 남기려고 여기 보관

    new_one_liner = None    # 보통 변수. 매 실행마다 None으로 시작 → "이번에 새 추천이 생겼나" 표시

    # ---------- 제목 ----------
    st.title(f"🎧 {APP_TITLE}")     # 큰 제목
    st.caption(APP_SUBTITLE)        # 작은 회색 설명
    st.markdown("---")              # 가로 구분선

    # ---------- 접히는 설명 ----------
    with st.expander("이 프로그램에 관하여", expanded=False):
        # ↑ with = "이 블록 안의 내용은 저 안에 넣어라". expanded=False → 처음엔 접힘
        st.write(
            """
            - UI는 **스트림릿(Streamlit)** 으로 만들었습니다.
            - 음성 인식(STT)과 노래 추천을 **Google Gemini** 가 한 번에 처리합니다.
              (교재는 Whisper + GPT를 따로 불러 API를 2번 호출합니다)
            - 답변 형식은 **구조화 출력(response_schema)** 으로 고정했습니다.
            - 한 줄 소개는 구글의 **Google Translate TTS(gTTS)** 로 읽어 줍니다.
            """
        )

    # ---------- 사이드바 (화면 왼쪽 고정 영역) ----------
    with st.sidebar:                          # 이 블록 안의 내용은 전부 왼쪽에 배치됨
        secret_key = load_default_apikey()    # 배포 환경의 Secrets에서 키를 읽어봄

        if secret_key:                        # 키가 있으면 (= 배포 환경이면)
            st.session_state["GEMINI_API"] = secret_key   # 서버 메모리에만 담아둠
            st.success("API 키가 설정되어 있습니다.", icon="🔑")   # 초록 상자로 알림
            st.caption("서버(Secrets)에 저장된 키를 사용합니다.")
            # ★ 여기서 입력창을 '만들지 않는' 것이 핵심입니다.
            #    st.text_input(value=키) 로 넣으면 그 값이 브라우저까지 전송돼서,
            #    type="password" 라도 눈 아이콘을 누르면 그대로 보입니다.
        else:                                 # 키가 없으면 (= 로컬이면)
            st.session_state["GEMINI_API"] = st.text_input(   # 직접 입력받는 칸을 만듦
                label="GEMINI API 키",        # 칸 위에 표시할 이름
                placeholder="Enter your API Key",   # 비어 있을 때 흐리게 보이는 안내
                value=st.session_state["GEMINI_API"],   # 처음에 채워둘 값
                type="password",              # 입력한 글자를 ●●● 로 가림
            )
            st.markdown(
                "[API 키 발급받기](https://aistudio.google.com/app/apikey)",
                help="Google AI Studio에서 무료로 발급받을 수 있습니다.",   # ? 아이콘 툴팁
            )
        st.markdown("---")

        model = st.radio(label="Gemini 모델", options=MODELS)
        # ↑ 라디오 버튼을 만들고, 선택된 값을 model 변수에 담음

        st.markdown("---")

        # ★ 초기화 버튼 — 이 프로젝트에서 제일 오래 붙잡은 부분
        if st.button(label="초기화"):          # 버튼을 만들고, 눌렸으면 True → 아래 실행
            st.session_state["chat"] = []      # 화면 기록 비움
            st.session_state["messages"] = []  # 대화 기록 비움
            st.session_state["last_audio_id"] = None   # 녹음 지문 비움
            st.session_state["last_error"] = None      # 오류 기록 비움
            st.session_state["audio_round"] += 1       # ★ 회차를 1 올림
            # ↑ 이 한 줄이 없으면:
            #   재실행 → 녹음 위젯에는 아까 녹음이 그대로 남아 있음
            #   → 지문이 None과 다르니 "새 녹음이네!" → 방금 지운 대화가 다시 생김
            #   회차를 올리면 위젯 이름표가 바뀌어서 스트림릿이 빈 위젯을 새로 만듭니다
            st.rerun()                         # 지금 실행을 멈추고 처음부터 다시 실행
            # ↑ 이게 없으면 아래쪽 화면이 '비우기 전 기록'으로 한 번 더 그려집니다

        st.caption(
            "예) 비 오는 날 듣기 좋은 노래 / 운동할 때 들을 신나는 곡 / "
            "새벽에 혼자 듣고 싶은 노래"
        )

    # ---------- 본문을 좌우 두 칸으로 나눔 ----------
    col1, col2 = st.columns([1, 1.4])   # 너비 비율 1 : 1.4 (오른쪽을 조금 더 넓게)

    with col1:                          # ===== 왼쪽 칸: 입력과 처리 =====
        st.subheader("말하기")           # 중간 크기 제목

        audio = st.audio_input(                                  # 녹음 위젯을 만듦
            "클릭하여 녹음하기",                                    # 위젯 위에 표시할 글
            key=f"audio_{st.session_state['audio_round']}",      # ★ 이름표에 회차를 붙임
            # ↑ key가 audio_0 → audio_1 로 바뀌면 스트림릿은 '다른 위젯'으로 보고
            #   빈 상태로 새로 만듭니다. 초기화 버튼이 제대로 동작하는 이유입니다
        )   # 녹음이 없으면 None, 있으면 파일 같은 객체를 돌려줌

        if audio is None:               # 녹음이 없을 때만 (= 마이크를 못 쓰는 사람을 위해)
            audio = st.file_uploader(                            # 파일 올리는 칸을 대신 보여줌
                "마이크가 안 되면 음성 파일을 올리세요 (wav, mp3)",
                type=["wav", "mp3", "ogg", "flac", "aac"],       # 허용할 확장자
                key=f"upload_{st.session_state['audio_round']}",  # 여기도 회차를 붙임
            )

        if audio is not None:           # 녹음이든 업로드든, 소리가 들어왔으면
            audio_bytes = audio.getvalue()   # 실제 소리 데이터를 꺼냄

            audio_id = hashlib.md5(audio_bytes, usedforsecurity=False).hexdigest()
            # ↑ 소리 데이터로 '지문'을 만듦. 내용이 같으면 지문도 같음
            #   usedforsecurity=False = "보안 용도가 아니라 비교용입니다" 라는 표시

            is_new_audio = audio_id != st.session_state["last_audio_id"]
            # ↑ 직전 지문과 다르면 True (= 새 녹음이다)

            # 아래는 if / elif 사슬. 위에서부터 검사해서 맞는 것 하나만 실행됩니다.

            if not audio_bytes:                     # ① 데이터가 비었으면
                st.error("빈 오디오입니다. 다시 녹음해 주세요.")   # 빨간 상자로 알림

            elif len(audio_bytes) > MAX_AUDIO_BYTES:   # ② 12MB를 넘으면
                st.error(
                    f"녹음이 너무 깁니다 ({len(audio_bytes) / 1048576:.1f}MB). "
                    # ↑ 1048576 = 1MB의 바이트 수. :.1f = 소수점 한 자리까지
                    f"{MAX_AUDIO_BYTES // 1048576}MB(약 6분) 이하로 해 주세요."
                    # ↑ // = 나눗셈 후 소수점 버림
                )

            elif not is_new_audio:                  # ③ 아까와 같은 녹음이면
                if st.session_state["chat"]:        # 대화가 하나라도 있을 때만 안내
                    st.caption(
                        "방금 처리한 것과 같은 녹음입니다. "
                        "새로 녹음하거나 사이드바에서 초기화해 주세요."
                    )
                    # ↑ 조용히 넘어가면 사용자는 앱이 멈춘 줄 압니다

            elif not st.session_state["GEMINI_API"]:   # ④ API 키가 없으면
                st.warning("사이드바에 Gemini API 키를 입력해 주세요.")   # 노란 상자

            elif len(st.session_state["chat"]) >= MAX_TURNS_PER_SESSION:  # ⑤ 10회를 넘었으면
                st.warning(
                    f"데모 앱이라 한 번에 {MAX_TURNS_PER_SESSION}곡까지만 추천합니다. "
                    "사이드바에서 초기화해 주세요."
                )

            else:                                   # ⑥ 위 검사를 전부 통과했으면 → 실제 처리
                st.session_state["last_audio_id"] = audio_id
                # ↑ API를 부르기 '전에' 지문을 기록합니다.
                #   실패하더라도 같은 녹음으로 무한 재시도하지 않도록

                with st.spinner("듣고 있어요... 오늘의 한 곡을 고르는 중입니다"):
                    # ↑ 이 블록이 끝날 때까지 화면에 빙글빙글 도는 표시를 띄움
                    try:
                        rec = recommend(                        # ★ 핵심 함수 호출
                            audio_bytes,                        # 소리 데이터
                            normalize_mime(audio.type),         # 형식 이름 정리해서
                            st.session_state["messages"],       # 지금까지의 대화
                            st.session_state["GEMINI_API"],     # API 키
                            model,                              # 선택한 모델
                        )
                        st.session_state["last_error"] = None   # 성공했으니 이전 오류 기록 지움
                    except Exception as e:                      # 실패하면
                        rec = None                              # 결과 없음으로 표시
                        st.session_state["last_error"] = str(e) # 오류 내용을 세션에 저장
                        # ↑ 여기서 st.error만 부르면 다음 재실행 때 메시지가 사라집니다

                if rec:                             # 결과를 제대로 받았으면
                    now = datetime.now(KST).strftime("%H:%M")
                    # ↑ 한국 시간으로 현재 시각을 "시:분" 형태 글자로. KST가 없으면 9시간 밀림

                    said = (rec.get("transcript") or "").strip()   # 받아쓴 문장 꺼내기

                    st.session_state["chat"].append(              # 화면 기록에 한 칸 추가
                        {"time": now, "said": said, "rec": rec}
                        # ↑ 말한 내용과 추천을 한 덩어리로 묶음 → 순서가 어긋나지 않음
                    )

                    st.session_state["messages"].append(          # 대화 기록에 사용자 발언 추가
                        {"role": "user", "content": said or "(무음)"}
                    )
                    st.session_state["messages"].append(          # 모델 발언도 추가
                        {
                            "role": "model",                      # Gemini에서는 assistant가 아니라 model
                            "content": f"{rec.get('title')} - {rec.get('artist')} 추천함",
                            # ↑ 전체 내용이 아니라 곡명만 남김 → 토큰 절약 + 같은 곡 재추천 방지
                        }
                    )

                    new_one_liner = rec.get("one_liner")   # "이번에 읽어줄 문장" 표시

        # ↓ 위 if 블록 '밖'입니다. 오디오가 없어도 오류는 계속 보여야 하니까요.
        if st.session_state["last_error"]:        # 저장된 오류가 있으면
            st.error(f"추천에 실패했습니다: {st.session_state['last_error']}")
            if st.button("같은 녹음으로 다시 시도"):   # 재시도 버튼
                st.session_state["last_audio_id"] = None   # 지문을 지워서 '새 녹음'으로 만듦
                st.session_state["last_error"] = None      # 오류 기록도 지움
                st.rerun()                                 # 처음부터 다시 실행 → 재처리됨

    with col2:                          # ===== 오른쪽 칸: 결과 보여주기 =====
        st.subheader("오늘의 추천")

        if not st.session_state["chat"]:    # 아직 대화가 하나도 없으면
            st.info("왼쪽에서 녹음 버튼을 누르고, 지금 기분이나 상황을 말해 보세요.")  # 파란 안내

        for entry in reversed(st.session_state["chat"]):   # 기록을 뒤에서부터 하나씩
            # ↑ reversed = 최근 것이 맨 위로 오게 뒤집음
            if entry["said"]:                              # 말한 내용이 있으면
                render_user_bubble(entry["said"], entry["time"])   # 파란 말풍선
            render_recommendation(entry["rec"], entry["time"])     # 추천 카드

        if new_one_liner:               # 이번 실행에서 새 추천이 생겼으면
            TTS(new_one_liner)          # 한 줄 소개만 음성으로 재생
            # ↑ 카드를 다 그린 '뒤'에 재생합니다. 글이 먼저 보이고 소리가 따라오도록
            #   그리고 new_one_liner는 매 실행 None으로 시작하므로,
            #   화면을 다시 그릴 때마다 이전 답변이 반복 재생되지 않습니다


# 이 파일을 직접 실행했을 때만 main()을 부릅니다.
# (다른 파일이 이 파일을 import 할 때는 실행되지 않음 → 테스트 코드가 쓰는 방식)
if __name__ == "__main__":
    main()

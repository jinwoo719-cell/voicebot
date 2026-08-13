"""
오늘 뭐 듣지? — 음성으로 오늘의 노래를 추천받는 프로그램

교재 <진짜 챗GPT API 활용법> PART 03 '나만의 음성 비서 만들기'를 바탕으로,
OpenAI(Whisper + GPT) 대신 Google Gemini API를 사용하고
'노래 추천'이라는 목적에 맞게 다시 만든 버전입니다.

  [교재]    녹음 -> STT(Whisper) -> 답변(GPT) -> TTS(gTTS) -> 재생   (API 2회)
  [본 코드] 녹음 -> 받아쓰기+추천을 Gemini 한 번에 -> TTS(gTTS)       (API 1회)

실행:  streamlit run voicebot.py
"""

##### 1. 패키지 불러오기 #####
import base64                       # mp3를 HTML에 심기 위한 인코딩
import hashlib                      # 같은 녹음을 중복 처리하지 않기 위한 해시
import html                         # 사용자 입력을 HTML에 넣을 때 이스케이프
import json                         # Gemini가 돌려준 JSON 파싱
import re                           # 한글 포함 여부 판별
from datetime import datetime       # 말풍선에 표시할 시각
from io import BytesIO              # gTTS 결과를 메모리에서 다루기 위함

import streamlit as st              # 웹 UI
from google import genai            # Gemini API
from google.genai import types      # 오디오/설정을 넘길 때 쓰는 타입
from gtts import gTTS               # TTS (Google Translate TTS)


##### 2. 상수 정의 #####
APP_TITLE = "오늘 뭐 듣지?"
APP_SUBTITLE = "말로 기분을 들려주세요. 오늘의 한 곡을 골라드립니다."

# 라디오 버튼에 노출할 Gemini 모델. 두 모델 모두 오디오 입력을 지원합니다.
MODELS = ["gemini-2.5-flash", "gemini-3.6-flash"]

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

# Gemini가 이 형태로만 답하도록 강제하는 스키마 (구조화 출력)
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


##### 3. 기능 구현 함수 #####
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


def recommend(audio_bytes: bytes, mime_type: str, history: list,
              apikey: str, model: str) -> dict:
    """오디오를 넘겨 '받아쓰기 + 노래 추천'을 한 번에 받아 옵니다.

    교재는 STT(Whisper)와 답변(GPT)을 따로 호출해서 왕복이 두 번이었습니다.
    Gemini는 오디오와 지시문을 같이 받을 수 있으므로 한 번에 끝냅니다.
    응답이 흐트러지지 않도록 response_schema로 형식을 강제합니다.
    """
    client = get_client(apikey)

    # 이전 대화를 먼저 넣어야 "아까랑 다른 걸로" 같은 후속 요청이 통합니다.
    contents = [
        types.Content(role=m["role"], parts=[types.Part(text=m["content"])])
        for m in history
    ]
    # 이번 차례: 지시문 + 오디오
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part(text=TASK_PROMPT),
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            ],
        )
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=RECOMMENDATION_SCHEMA,
        ),
    )

    # 스키마를 강제했어도 안전 필터 등으로 빈 응답이 올 수 있습니다.
    raw = (response.text or "").strip()
    if not raw:
        raise ValueError("Gemini가 빈 응답을 돌려주었습니다.")
    return json.loads(raw)


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

    st.markdown(
        f"""
        <div style="border:1px solid #e3e6ea;border-radius:14px;padding:18px 20px;
                    background:#fafbfc;margin-bottom:1.2rem;">
          <div style="font-size:1.25rem;font-weight:700;line-height:1.3;">
            🎵 {title}
          </div>
          <div style="color:#6b7280;font-size:0.95rem;margin-top:2px;">{artist}</div>
          <div style="margin-top:12px;padding:10px 14px;background:#eef4ff;
                      border-radius:10px;font-size:0.95rem;">
            {one_liner}
          </div>
          <div style="margin-top:16px;">
            <div style="font-size:0.75rem;letter-spacing:0.04em;color:#6b7280;
                        font-weight:700;">추천 이유</div>
            <div style="margin-top:4px;line-height:1.65;">{reason}</div>
          </div>
          <div style="margin-top:14px;">
            <div style="font-size:0.75rem;letter-spacing:0.04em;color:#6b7280;
                        font-weight:700;">노래 배경</div>
            <div style="margin-top:4px;line-height:1.65;">{background}</div>
          </div>
          <div style="text-align:right;font-size:0.8rem;color:gray;margin-top:10px;">
            {html.escape(time)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


##### 4. 메인 함수 #####
def main():
    # --- 기본 설정 ---
    st.set_page_config(page_title=APP_TITLE, page_icon="🎧", layout="wide")

    # --- 세션 상태 초기화 ---
    # 화면에 그릴 기록: {"time": ..., "said": ..., "rec": {...}}
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

        # 초기화 버튼
        if st.button(label="초기화"):
            st.session_state["chat"] = []
            st.session_state["messages"] = []
            st.session_state["last_audio_id"] = None
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
            audio_id = hashlib.md5(audio_bytes).hexdigest()
            is_new_audio = audio_id != st.session_state["last_audio_id"]

            if is_new_audio and not st.session_state["GEMINI_API"]:
                st.warning("사이드바에 Gemini API 키를 입력해 주세요.")

            elif is_new_audio:
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
                    except Exception as e:
                        rec = None
                        st.error(f"추천에 실패했습니다: {e}")

                if rec:
                    now = datetime.now().strftime("%H:%M")
                    said = (rec.get("transcript") or "").strip()

                    # 화면에 그릴 기록. 한 번의 대화를 한 덩어리로 묶어 둡니다.
                    # (말한 내용과 추천이 항상 짝을 이뤄 순서가 어긋나지 않습니다)
                    st.session_state["chat"].append(
                        {"time": now, "said": said, "rec": rec}
                    )

                    # 다음 요청에 문맥이 이어지도록 대화 기록에도 저장
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

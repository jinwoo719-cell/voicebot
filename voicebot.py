"""
음성 비서 프로그램 (Gemini 버전)

교재 <진짜 챗GPT API 활용법> PART 03 '나만의 음성 비서 만들기'를
OpenAI(Whisper + GPT) 대신 Google Gemini API로 구현한 버전입니다.

  [교재]  녹음(audiorecorder) -> STT(Whisper) -> 답변(GPT) -> TTS(gTTS) -> 재생
  [본 코드] 녹음(st.audio_input) -> STT(Gemini) -> 답변(Gemini) -> TTS(gTTS) -> 재생

실행:  streamlit run voicebot.py
"""

##### 1. 패키지 불러오기 #####
import base64                       # 음원 파일을 HTML에 심기 위한 인코딩
import hashlib                      # 같은 녹음을 중복 처리하지 않기 위한 해시
import re                           # 답변 언어(한/영) 판별
from datetime import datetime       # 채팅 말풍선에 표시할 시각
from io import BytesIO              # gTTS 결과를 메모리에서 다루기 위함

import streamlit as st              # 웹 UI
from google import genai            # Gemini API (STT + 답변)
from google.genai import types      # Gemini에 오디오/설정을 넘길 때 쓰는 타입
from gtts import gTTS               # TTS (Google Translate TTS)


##### 2. 상수 정의 #####
# 라디오 버튼에 노출할 Gemini 모델. 두 모델 모두 오디오 입력을 지원합니다.
MODELS = ["gemini-3.6-flash", "gemini-2.5-flash"]

# 답변 스타일을 지정하는 시스템 프롬프트 (교재의 role: "system" 메시지에 해당)
SYSTEM_PROMPT = (
    "You are a thoughtful assistant. Respond to all input in 25 words or less, "
    "and answer in the same language the user used."
)

# 음성을 텍스트로 받아쓰게 하는 지시문. '받아쓴 문장만' 내놓도록 못 박습니다.
STT_PROMPT = (
    "Transcribe the following audio to text exactly as spoken. "
    "Return ONLY the transcript itself, with no explanation, no quotation marks, "
    "and no extra commentary. If the audio contains no discernible speech, "
    "return an empty string."
)


##### 3. 기능 구현 함수 #####
def load_default_apikey() -> str:
    """배포 환경(Streamlit Cloud)의 Secrets에 키가 있으면 기본값으로 씁니다.

    로컬에는 보통 .streamlit/secrets.toml이 없고, 그 상태로 st.secrets에 접근하면
    StreamlitSecretNotFoundError가 나서 앱이 죽습니다. 그래서 감싸 줍니다.
    """
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


def get_client(apikey: str) -> genai.Client:
    """API 키로 Gemini 클라이언트를 생성합니다. (교재의 openai.OpenAI()에 해당)"""
    return genai.Client(api_key=apikey)


def STT(audio_bytes: bytes, mime_type: str, apikey: str, model: str) -> str:
    """음성 -> 텍스트. 교재는 Whisper API를 썼지만 여기서는 Gemini가 직접 오디오를 듣습니다.

    Whisper와 달리 Gemini는 파일 경로가 아니라 바이트를 그대로 받으므로
    임시 파일 생성/삭제(교재 14~15, 24행)가 필요 없습니다.
    """
    client = get_client(apikey)
    response = client.models.generate_content(
        model=model,
        contents=[
            STT_PROMPT,
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
        ],
    )
    return (response.text or "").strip()


def ask_gemini(messages: list, model: str, apikey: str) -> str:
    """대화 기록을 통째로 넘겨 답변을 생성합니다. (교재의 ask_gpt()에 해당)

    messages 는 [{"role": "user"|"model", "content": "..."}] 형태입니다.
    OpenAI는 assistant, Gemini는 model 이라는 역할 이름을 씁니다.
    """
    client = get_client(apikey)

    # 우리 형식(dict) -> Gemini 형식(types.Content)으로 변환
    contents = [
        types.Content(role=m["role"], parts=[types.Part(text=m["content"])])
        for m in messages
    ]

    response = client.models.generate_content(
        model=model,
        contents=contents,
        # 시스템 프롬프트는 대화 기록이 아니라 config로 분리해서 전달합니다.
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return (response.text or "").strip()


def TTS(response: str) -> None:
    """텍스트 -> 음성. 교재와 동일하게 gTTS를 쓰되, 디스크 대신 메모리에서 처리합니다."""
    # 한글이 하나라도 있으면 한국어로, 아니면 영어로 읽습니다.
    lang = "ko" if re.search(r"[가-힣]", response) else "en"

    # 음성 파일 생성 (교재 19~21행: filename에 저장 -> 여기서는 BytesIO 버퍼)
    # gTTS는 구글 번역 서버를 호출하므로 실패할 수 있습니다.
    # 소리가 안 나는 것보다 답변 텍스트가 사라지는 게 더 큰 문제라, 여기서 막습니다.
    try:
        buffer = BytesIO()
        gTTS(text=response, lang=lang).write_to_fp(buffer)
        audio_bytes = buffer.getvalue()
    except Exception as e:
        st.warning(f"음성 변환에 실패했습니다(답변은 위에 표시됩니다): {e}")
        return

    # 음원 파일 자동 재생 (교재 24~32행과 동일한 HTML autoplay 방식)
    b64 = base64.b64encode(audio_bytes).decode()
    md = f"""
        <audio autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)

    # 브라우저가 자동 재생을 막았을 때를 대비한 수동 재생 플레이어
    st.audio(audio_bytes, format="audio/mp3")


##### 4. 메인 함수 #####
def main():
    # --- 기본 설정 ---
    st.set_page_config(page_title="음성 비서 프로그램", layout="wide")

    # --- 세션 상태 초기화 ---
    # 화면에 그릴 채팅 기록: (sender, time, message) 튜플의 목록
    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    # Gemini에 넘길 대화 기록: role/content 딕셔너리의 목록
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 사이드바에 입력한 API 키
    if "GEMINI_API" not in st.session_state:
        # 배포 환경에서는 Secrets에 넣어둔 키를 기본값으로 사용합니다.
        st.session_state["GEMINI_API"] = load_default_apikey()

    # 같은 녹음을 다시 처리하지 않기 위한 직전 녹음의 지문(해시)
    if "last_audio_id" not in st.session_state:
        st.session_state["last_audio_id"] = None

    # 녹음 위젯의 key에 붙일 회차 번호.
    # 초기화할 때 이 번호를 올리면 위젯이 새것으로 교체되어 녹음이 비워집니다.
    if "audio_round" not in st.session_state:
        st.session_state["audio_round"] = 0

    # 이번 실행에서 새로 만들어진 답변 (있으면 마지막에 음성으로 읽어줍니다)
    new_response = None

    # --- 제목 ---
    st.header("음성 비서 프로그램")
    st.markdown("---")

    # --- 기본 설명 ---
    with st.expander("음성비서 프로그램에 관하여", expanded=True):
        st.write(
            """
            - 음성 비서 프로그램의 UI는 **스트림릿(Streamlit)** 을 활용했습니다.
            - STT(Speech-To-Text)는 **Google Gemini** 의 오디오 이해 기능을 활용했습니다.
            - 답변은 **Google Gemini** 모델을 활용했습니다.
            - TTS(Text-To-Speech)는 구글의 **Google Translate TTS(gTTS)** 를 활용했습니다.
            """
        )
        st.markdown("")

    # --- 사이드바 ---
    with st.sidebar:
        # API 키 입력
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

    # --- 기능 구현 공간 ---
    col1, col2 = st.columns(2)

    with col1:
        # 왼쪽 영역: 질문하기
        st.subheader("질문하기")

        # 스트림릿 내장 녹음 위젯. 16kHz WAV 바이트를 돌려줍니다.
        # key에 회차 번호를 넣어, 초기화할 때 위젯 자체가 비워지도록 합니다.
        audio = st.audio_input(
            "클릭하여 녹음하기",
            key=f"audio_{st.session_state['audio_round']}",
        )

        if audio is not None:
            audio_bytes = audio.getvalue()
            # 녹음 내용의 해시를 지문으로 삼아, 새 녹음일 때만 처리합니다.
            audio_id = hashlib.md5(audio_bytes).hexdigest()
            is_new_audio = audio_id != st.session_state["last_audio_id"]

            if is_new_audio and not st.session_state["GEMINI_API"]:
                st.warning("사이드바에 Gemini API 키를 입력해 주세요.")

            elif is_new_audio:
                st.session_state["last_audio_id"] = audio_id

                # 음원 파일에서 텍스트 추출
                with st.spinner("음성을 텍스트로 변환하는 중입니다..."):
                    try:
                        question = STT(
                            audio_bytes,
                            audio.type or "audio/wav",
                            st.session_state["GEMINI_API"],
                            model,
                        )
                    except Exception as e:
                        question = ""
                        st.error(f"음성 인식에 실패했습니다: {e}")

                if question:
                    # 채팅을 시각화하기 위해 질문 내용 저장
                    now = datetime.now().strftime("%H:%M")
                    st.session_state["chat"].append(("user", now, question))
                    # 다음 질문에 문맥이 이어지도록 대화 기록에도 저장
                    st.session_state["messages"].append(
                        {"role": "user", "content": question}
                    )

                    # Gemini에게 답변 얻기
                    with st.spinner("답변을 생성하는 중입니다..."):
                        try:
                            response = ask_gemini(
                                st.session_state["messages"],
                                model,
                                st.session_state["GEMINI_API"],
                            )
                        except Exception as e:
                            response = ""
                            st.error(f"답변 생성에 실패했습니다: {e}")

                    if response:
                        # 후속 질문에 대비해 답변도 대화 기록에 저장
                        # (Gemini에서 모델의 발화는 role이 "model" 입니다)
                        st.session_state["messages"].append(
                            {"role": "model", "content": response}
                        )
                        now = datetime.now().strftime("%H:%M")
                        st.session_state["chat"].append(("bot", now, response))
                        new_response = response

    with col2:
        # 오른쪽 영역: 질문/답변
        st.subheader("질문/답변")

        # 채팅 형식으로 시각화하기
        for sender, time, message in st.session_state["chat"]:
            if sender == "user":
                st.write(
                    f'<div style="display:flex;align-items:center;">'
                    f'<div style="background-color:#007AFF;color:white;border-radius:12px;'
                    f'padding:8px 12px;margin-right:8px;">{message}</div>'
                    f'<div style="font-size:0.8rem;color:gray;">{time}</div></div>',
                    unsafe_allow_html=True,
                )
                st.write("")
            else:
                st.write(
                    f'<div style="display:flex;align-items:center;justify-content:flex-end;">'
                    f'<div style="background-color:lightgray;border-radius:12px;'
                    f'padding:8px 12px;margin-left:8px;">{message}</div>'
                    f'<div style="font-size:0.8rem;color:gray;">{time}</div></div>',
                    unsafe_allow_html=True,
                )
                st.write("")

        # gTTS를 활용하여 음성 파일 생성 및 재생
        # (이번 실행에서 새로 만들어진 답변이 있을 때만)
        if new_response:
            TTS(new_response)


if __name__ == "__main__":
    main()

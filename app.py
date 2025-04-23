import streamlit as st
import requests
import json
import uuid
from typing import List, Dict
import time
import os
import logging

# Configure logging
logging.basicConfig(filename="../backend/streamlit_app.log", level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Streamlit page configuration
st.set_page_config(page_title="小金毛旅游导航", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "cities" not in st.session_state:
    st.session_state.cities = []
if "mode" not in st.session_state:
    st.session_state.mode = None
if "plan" not in st.session_state:
    st.session_state.plan = None
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

# API endpoint
API_URL = "http://localhost:8001/api/plan/stream"

# Check for static files
GOLDEN_RETRIEVER_PATH = "../static/Golden_retriever.png"
FALLBACK_GOLDEN_RETRIEVER_URL = "https://img.icons8.com/color/48/000000/dog.png"
if not os.path.exists(GOLDEN_RETRIEVER_PATH):
    st.warning(f"未找到 {GOLDEN_RETRIEVER_PATH}，使用在线金毛头像")
    GOLDEN_RETRIEVER_SRC = FALLBACK_GOLDEN_RETRIEVER_URL
else:
    GOLDEN_RETRIEVER_SRC = GOLDEN_RETRIEVER_PATH

USER_AVATAR_PATH = "../static/user_avatar.png"
FALLBACK_USER_AVATAR_URL = "https://img.icons8.com/color/48/000000/person-male.png"
if not os.path.exists(USER_AVATAR_PATH):
    st.warning(f"未找到 {USER_AVATAR_PATH}，使用在线用户头像")
    USER_AVATAR_SRC = FALLBACK_USER_AVATAR_URL
else:
    USER_AVATAR_SRC = USER_AVATAR_PATH

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #FFF8E7;
        font-family: 'Arial Rounded MT Bold', 'Comic Sans MS', sans-serif;
    }
    .stButton>button {
        background-color: #FFC107;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FFA000;
        transform: scale(1.05);
    }
    .chat-container {
        display: flex;
        align-items: flex-start;
        margin: 10px 0;
    }
    .user-container {
        flex-direction: row-reverse;
    }
    .bot-container {
        flex-direction: row;
    }
    .chat-message {
        padding: 12px;
        border-radius: 12px;
        max-width: 70%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #DCF8C6;
        margin-left: 10px;
    }
    .bot-message {
        background-color: #FFF3CD;
        margin-right: 10px;
    }
    .avatar {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid #FFC107;
        background-color: #FFFFFF;
    }
    .thinking-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #FFFFFF;
        margin: 10px 0;
        max-width: 80%;
        border: 2px solid #FFE082;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .thinking-header {
        font-weight: bold;
        color: #5D4037;
        margin-bottom: 10px;
    }
    .final-summary {
        background-color: #FFF8E7;
        border: 2px solid #FFC107;
        padding: 15px;
        border-radius: 12px;
    }
    .stTextInput>div>input {
        border-radius: 8px;
        border: 2px solid #FFC107;
    }
    .stTextArea>div>textarea {
        border-radius: 8px;
        border: 2px solid #FFC107;
    }
    .stSlider>div {
        padding: 10px 0;
    }
    .sidebar .sidebar-content {
        background-color: #FFF3CD;
    }
    h1, h2, h3 {
        color: #5D4037;
    }
    </style>
""", unsafe_allow_html=True)

def add_to_chat_history(role: str, content: str):
    """Add message to chat history"""
    st.session_state.chat_history.append({"role": role, "content": content})

def clear_cities():
    """Clear city list"""
    st.session_state.cities = []

def submit_plan(mode: str, user_input: str):
    """Submit travel plan request to backend, stream partial results as thinking process, and display final summary"""
    try:
        payload = {
            "mode": mode,
            "userInput": user_input,
            "conversation_id": st.session_state.conversation_id
        }

        if mode == "single":
            if not st.session_state.cities:
                st.error("请为单城市模式添加一个城市。")
                return None
            payload["city"] = st.session_state.cities[0]["name"]
            payload["days"] = st.session_state.cities[0]["days"]
        else:
            if not st.session_state.cities:
                st.error("请为多城市模式添加至少一个城市。")
                return None
            payload["cities"] = st.session_state.cities

        logging.info(f"Sending request to backend: {payload}")
        start_time = time.time()
        logging.debug(f"Request started at: {start_time}")

        thinking_box = st.empty()
        thinking_content = '<div class="thinking-header">小金毛的规划过程</div>\n\n'
        final_content = ""
        thinking_box.markdown(f'<div class="thinking-box">{thinking_content}</div>', unsafe_allow_html=True)

        with requests.post(API_URL, json=payload, stream=True, timeout=300) as response:
            response.raise_for_status()
            partial_plan = {}
            complete_plan = None
            key_translations = {
                "weather": "天气信息",
                "scenic_spots": "景点安排",
                "dining": "餐饮推荐",
                "accommodation": "住宿安排",
                "transportation": "交通规划",
                "summary": "分项总结",
                "final_summary": "最终总结"
            }

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        logging.debug(f"Backend response line: {data}")
                        if "partial" in data:
                            for key, value in data["partial"].items():
                                partial_plan[key] = value
                                if key != "final_summary":
                                    display_key = key_translations.get(key, key)
                                    new_content = f"**{display_key}**：{value}\n\n"
                                    thinking_content += new_content
                                    thinking_box.markdown(f'<div class="thinking-box">{thinking_content}</div>', unsafe_allow_html=True)
                                else:
                                    # Store final_summary temporarily
                                    partial_plan["final_summary"] = value
                        elif "complete" in data:
                            complete_plan = data["complete"]
                            if complete_plan is None:
                                logging.error("Received None as complete_plan")
                                st.error("后端返回的完整计划为空，请检查输入或稍后重试。")
                                thinking_box.empty()
                                return None
                            st.session_state.plan = {"mode": mode, "plan": complete_plan}
                            break
                        elif "error" in data:
                            st.error(f"后端错误：{data['error']}")
                            logging.error(f"Backend error: {data['error']}")
                            thinking_box.empty()
                            return None
                    except json.JSONDecodeError as e:
                        st.error(f"解析后端响应失败：{str(e)}")
                        logging.error(f"JSON decode error: {str(e)}, Raw line: {line.decode('utf-8')}")
                        thinking_box.empty()
                        return None

            thinking_box.empty()
            end_time = time.time()
            logging.info(f"Request completed in {end_time - start_time:.2f} seconds")

            if complete_plan is None:
                logging.error("No complete plan received from backend")
                st.error("未收到后端的完整计划，请检查输入或稍后重试。")
                return None

            # Construct final content
            if mode == "multi":
                final_content = ""
                for city_plan in complete_plan:
                    if not isinstance(city_plan, dict):
                        logging.error(f"Invalid city_plan format: {city_plan}")
                        continue
                    city = city_plan.get("city", "未知城市")
                    days = city_plan.get("days", 1)
                    final_content += f"### {city} ({days} 天)\n"
                    final_content += f"{city_plan.get('final_summary', '暂无总结')}\n\n"
            else:
                city = complete_plan.get("city", "未知城市")
                days = complete_plan.get("days", 1)
                final_content = f"### {city} ({days} 天)\n"
                final_content += f"{partial_plan.get('final_summary', '暂无总结')}\n\n"

            return final_content

    except requests.exceptions.RequestException as e:
        st.error(f"无法连接到后端：{str(e)}")
        logging.error(f"Request error: {str(e)}", exc_info=True)
        thinking_box.empty()
        return None

# Sidebar for mode selection and city input
with st.sidebar:
    st.header("小金毛导航设置")
    st.image(GOLDEN_RETRIEVER_SRC, width=100)

    mode = st.radio("选择旅行模式", ["单城市", "多城市"], key="mode_selection")
    st.session_state.mode = "single" if mode == "单城市" else "multi"

    with st.form(key="city_form"):
        st.subheader("添加目的地")
        city_name = st.text_input("城市名称", placeholder="例如：北京")
        days = st.slider("旅行天数", min_value=1, max_value=30, value=3)
        submit_city = st.form_submit_button("添加城市")

        if submit_city and city_name:
            st.session_state.cities.append({"name": city_name, "days": days})
            st.success(f"小金毛已添加 {city_name}，旅行 {days} 天！")

    if st.session_state.cities:
        st.subheader("已选目的地")
        for i, city in enumerate(st.session_state.cities):
            st.write(f"{city['name']} - {city['days']} 天")

        if st.button("清空城市"):
            clear_cities()
            st.success("小金毛已清空所有城市！")

# Main content area
st.title("🐶 小金毛旅游导航")
st.markdown("跟随小金毛，规划您的完美旅行！")

chat_container = st.container()
input_container = st.container()

with chat_container:
    st.subheader("与小金毛的对话")
    for message in st.session_state.chat_history:
        with st.container():
            if message["role"] == "user":
                st.markdown(
                    f"""
                    <div class="chat-container user-container">
                        <img src="{USER_AVATAR_SRC}" class="avatar">
                        <div class="chat-message user-message">{message["content"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="chat-container bot-container">
                        <img src="{GOLDEN_RETRIEVER_SRC}" class="avatar">
                        <div class="chat-message bot-message final-summary">{message["content"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

with input_container:
    with st.form(key="chat_form"):
        user_input = st.text_area("告诉小金毛您的需求", placeholder="例如：我想体验北京的文化之旅，或查询上海的天气信息", height=100)
        col1, col2 = st.columns([1, 1])
        with col1:
            submit_button = st.form_submit_button("提交请求")
        with col2:
            clear_button = st.form_submit_button("清空对话")

        if submit_button and user_input:
            add_to_chat_history("user", user_input)
            with st.spinner("小金毛正在为您规划..."):
                response = submit_plan(st.session_state.mode, user_input)
                if response:
                    add_to_chat_history("assistant", response)
                    st.rerun()

        if clear_button:
            st.session_state.chat_history = []
            st.session_state.plan = None
            st.session_state.conversation_id = str(uuid.uuid4())
            st.rerun()

# Footer
st.markdown("---")
st.markdown("© 2025 小金毛旅游导航")
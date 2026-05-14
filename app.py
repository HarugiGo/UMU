import streamlit as st
import openai
import pandas as pd
import json
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="语音关键词分析", layout="wide")
st.title("语音关键词分析工具")

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

left, right = st.columns([1, 1.5])

# ── 左侧 ──────────────────────────────────────
with left:
    st.subheader("🎙️ 上传录音")
    audio_file = st.file_uploader(
        "支持 MP4 / MP3 / WAV / M4A（≤30分钟）",
        type=["mp4", "mp3", "wav", "m4a"]
    )

    st.subheader("🔑 关键词输入")
    method = st.radio("输入方式", ["手动输入", "Excel 上传"], horizontal=True)

    keywords = []
    if method == "手动输入":
        raw = st.text_area(
            "每行一个关键词（最多100个，每词≤10字）",
            height=220,
            placeholder="销售额\n客户满意度\n市场占有率"
        )
        if raw:
            keywords = [k.strip() for k in raw.splitlines() if k.strip()]
    else:
        excel = st.file_uploader("上传 Excel 文件", type=["xlsx", "xls"], key="excel")
        if excel:
            df_kw = pd.read_excel(excel, header=None)
            keywords = df_kw.iloc[:, 0].dropna().astype(str).tolist()
            st.success(f"已读取 {len(keywords)} 个关键词")
            st.dataframe(pd.DataFrame(keywords, columns=["关键词"]), height=180)

    keywords = [k for k in keywords if len(k) <= 10][:100]
    if keywords:
        st.caption(f"有效关键词：{len(keywords)} 个")

    run = st.button("🚀 开始分析", type="primary",
                    disabled=not (audio_file and keywords))

# ── 右侧 ──────────────────────────────────────
with right:
    if run:
        # Step 1: 转录
        with st.spinner("转录中，请稍候..."):
            audio_bytes = audio_file.read()
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=(audio_file.name, audio_bytes),
                response_format="text"
            )
            transcript = resp

        # Step 2: 摘要
        with st.spinner("生成摘要..."):
            summary_resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "请用不超过150字概括以下转录内容的核心要点。"},
                    {"role": "user", "content": transcript}
                ]
            )
            summary = summary_resp.choices[0].message.content

        st.subheader("📋 内容摘要")
        st.info(summary)
        st.download_button(
            "⬇️ 下载完整转录文本 (TXT)",
            data=transcript.encode("utf-8"),
            file_name="transcript.txt",
            mime="text/plain"
        )

        st.divider()

        # Step 3: 直接触达（字符串匹配）
        direct = {kw: kw in transcript for kw in keywords}

        # Step 4: 语义触达（LLM 批处理）
        with st.spinner("语义分析中..."):
            kw_numbered = "\n".join(f"{i+1}. {kw}" for i, kw in enumerate(keywords))
            sem_resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": (
                        "判断转录文本是否在语义上提及了每个关键词（即使用了不同表达）。"
                        "返回JSON格式：{\"关键词\": true或false}"
                    )},
                    {"role": "user", "content":
                        f"转录文本：\n{transcript}\n\n关键词列表：\n{kw_numbered}"}
                ],
                response_format={"type": "json_object"}
            )
            semantic = json.loads(sem_resp.choices[0].message.content)

        # 结果表格
        st.subheader("📊 关键词分析结果")
        rows = [{
            "关键词": kw,
            "直接触达": "✅ 是" if direct.get(kw) else "❌ 否",
            "语义触达": "✅ 是" if semantic.get(kw) else "❌ 否"
        } for kw in keywords]

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # 计分
        d_score = sum(direct.values())
        s_score = sum(1 for v in semantic.values() if v)
        total = len(keywords)

        st.subheader("🏆 得分")
        c1, c2 = st.columns(2)
        c1.metric("直接触达", f"{d_score} / {total}", f"{d_score/total*100:.1f}%")
        c2.metric("语义触达", f"{s_score} / {total}", f"{s_score/total*100:.1f}%")

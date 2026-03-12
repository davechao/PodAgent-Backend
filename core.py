import os
import xml.etree.ElementTree as ET
import requests
import subprocess
import json

from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from pinecone.exceptions import NotFoundException

from openai import OpenAI
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.messages import HumanMessage

load_dotenv()

TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)

# 初始化 OpenAI
client = OpenAI()

# 初始化 Chat LLM
llm = init_chat_model(
    model="gpt-4",
    model_provider="openai"
)

# 初始化 Embeddings + VectorStore
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    chunk_size=100,
    show_progress_bar=False
)
vectorstore = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"),
    embedding=embeddings
)


# --------------------------------
#  從 RSS 取得 Podcast
# --------------------------------
def get_podcast_info(rss_url: str):
    resp = requests.get(rss_url)
    root = ET.fromstring(resp.content)

    # podcast 名稱
    channel = root.find("channel")
    podcast_title = channel.find("title").text if channel is not None else "Unknown Podcast"

    episodes = []

    for item in root.findall(".//item"):
        title = item.find("title")
        pub_date_tag = item.find("pubDate")
        enclosure = item.find("enclosure")

        title = title.text if title is not None else "No Title"

        pub_date = None
        if pub_date_tag is not None:
            try:
                dt = datetime.strptime(pub_date_tag.text, "%a, %d %b %Y %H:%M:%S %Z")
                pub_date = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                pub_date = pub_date_tag.text

        mp3_url = enclosure.get("url") if enclosure is not None else None

        episodes.append({
            "title": title,
            "publish_time": pub_date,
            "mp3_url": mp3_url
        })

    return {
        "podcast_title": podcast_title,
        "episodes": episodes
    }


# --------------------------------
#  下載 MP3 + FFmpeg 切段
# --------------------------------
def download_and_split_mp3(mp3_url: str):
    episode_dir = TMP_DIR / f"episode"
    episode_dir.mkdir(exist_ok=True)

    audio_path = episode_dir / "podcast.mp3"

    print("Downloading:", mp3_url)

    audio = requests.get(mp3_url).content

    with open(audio_path, "wb") as f:
        f.write(audio)

    print("Splitting audio with ffmpeg...")

    cmd = [
        "ffmpeg",
        "-i", str(audio_path),
        "-f", "segment",
        "-segment_time", "60",
        "-c", "copy",
        str(episode_dir / "chunk_%03d.mp3")
    ]

    subprocess.run(cmd)

    files = sorted(episode_dir.glob("chunk_*.mp3"))

    return files


# --------------------------------
#  Whisper STT
# --------------------------------
def whisper_stt_from_episode():
    episode_dir = TMP_DIR / f"episode"

    files = sorted(episode_dir.glob("chunk_*.mp3"))

    transcripts = []

    for file in files:
        print("Transcribing:", file)

        with open(file, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )

        transcripts.append(transcript.text)

    full_text = "\n".join(transcripts)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([full_text])

    return docs


# --------------------------------
#  存入 VectorStore Pinecone
# --------------------------------
def save_to_pinecone(docs):
    print("Saving to Pinecone...")
    vectorstore.add_documents(docs)
    print("Saved", len(docs), "documents")


# --------------------------------
#  清除 VectorStore Pinecone Index
# --------------------------------
def clear_pinecone():
    print("Clearing Pinecone VectorStore...")
    try:
        vectorstore.delete(delete_all=True)
    except NotFoundException:
        print("Namespace not found, skip delete.")
    print("VectorStore cleared!")


# ------------------------------------
#  從 VectorStore Pinecone 查詢相似內容
# ------------------------------------
def get_episode_chunks(k: int = 50) -> list[str]:
    docs = vectorstore.similarity_search(query=" ", k=k)
    return [doc.page_content for doc in docs]


# ---------------------------
#  Run Agent
# ---------------------------
def run_agent(k: int = 50):
    docs = get_episode_chunks(k)

    prompt = f"""
你是一個整理 Podcast Episode 的助手，請閱讀下列內容，產生摘要與重點，並回傳 JSON：
{{
    "summary": "...", 
    "key_points": ["...", "..."]
}}

Episode 內容：
""" + "\n".join(docs)

    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {"summary": text, "key_points": []}

    return {
        "answer": result.get("summary"),
        "key_points": result.get("key_points"),
    }
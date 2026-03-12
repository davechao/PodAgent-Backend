from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from urllib.parse import unquote

from core import get_podcast_info, download_and_split_mp3, whisper_stt_from_episode, save_to_pinecone, clear_pinecone, \
    run_agent

app = FastAPI(title="Podcast API")


class Episode(BaseModel):
    title: str
    publish_time: str
    mp3_url: str


class Podcast(BaseModel):
    podcast_title: str
    episodes: List[Episode]


class Summary(BaseModel):
    answer: str
    key_points: list[str]


@app.get("/podcast", response_model=Podcast)
def api_get_podcast(rss_url: str):
    podcast = get_podcast_info(rss_url)
    return podcast


@app.post("/analyze", response_model=Summary)
def api_analyze_episode(mp3_url: str):
    try:
        clear_pinecone()

        download_and_split_mp3(unquote(mp3_url))
        docs = whisper_stt_from_episode()
        save_to_pinecone(docs)

        summary = run_agent("幫我整理這集Episode，列出重點與摘要")
        return summary

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

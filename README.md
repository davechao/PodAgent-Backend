# PodAgent

> AI 幫你「聽 Podcast → 整理 → 摘要 → 問答」，使用者不需要真的聽完整集。

---

## ✨ 功能
- 📡 輸入 RSS URL，自動列出Episodes
- ⬇️ 下載 MP3 音訊，並透過 FFmpeg 切段
- 🎙️ Whisper STT 語音轉文字
- 🗂️ 向量化後存入 Pinecone VectorStore
- 🤖 LLM Agent 自動產生摘要與重點

---

## 🏗️ 系統架構

```
RSS Feed URL
     │
     ▼
┌─────────────────────┐
│   RSS Parser        │  取得 Episode 資訊
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│   Audio Downloader  │  下載 MP3
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│   FFmpeg Chunker    │  切成1分鐘Chunk，降低 Whisper 記憶體用量
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│   Whisper STT       │  openai-whisper — 語音轉文字
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│   Embedding         │  OpenAI text-embedding-3-small
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│   Pinecone          │  將向量存入 VectorStore
└─────────────────────┘
     │
     ├─── 摘要流程
     │                                                           
     ▼                                                           
┌─────────────────────┐                             
│  LangChain Agent    │                             
│  LLM 產生摘要重點     │                             
└─────────────────────┘                             
     │                                               
     ▼
  摘要 + 重點列表                                         
```

---
## 🛠️ 技術棧
| 用途 | 套件                            |
|------|-------------------------------|
| API Server | FastAPI                       |
| 音訊切段 | FFmpeg                        |
| 語音轉文字 | openai-whisper                |
| Embedding | OpenAI text-embedding-3-small |
| Vector Store | Pinecone                      |
| LLM / Agent | LangChain + OpenAI GPT-4      |

#!/usr/bin/env python
# coding: utf-8

# # Replication - comparing sources
# 
# Changes:
# - Only 3 search queries per question
# - Only 1 date range (between begin date and a week prior to resolution)
# - Fetch articles from any source
# - Calculate probabilities separately (per source)
# - Compare Brier score between sources

# ## 0. Configuration

# In[ ]:


# ============================================================
# CELL 0 — Configuration
# This is the only cell you need to edit to change models, paths, or experiment settings.
# ============================================================

import os, sys

# ── Number of questions to evaluate ─────────────────────────
NUM_QUESTIONS = 50 # 840 total

# ── Models ──────────────────────────────────────────────────
# All LLM calls go through this model unless overridden below.
DEFAULT_MODEL = "llama-3.3-70b-versatile"

SEARCH_QUERY_MODEL    = DEFAULT_MODEL
RANKING_MODEL         = DEFAULT_MODEL
SUMMARIZATION_MODEL   = DEFAULT_MODEL
REASONING_MODEL       = DEFAULT_MODEL

# ── LLM Backend ─────────────────────────────────────────────
USE_LOCAL_LLM = True   # Set to False to use Groq instead

# ── Temperatures ────────────────────────────────────────────
SEARCH_QUERY_TEMPERATURE  = 0.0
RANKING_TEMPERATURE       = 0.0
SUMMARIZATION_TEMPERATURE = 0.2
REASONING_TEMPERATURE     = 1.0

# ── Retrieval settings ───────────────────────────────────────
RETRIEVAL_DAYS_BEFORE_RESOLUTION = 7
NUM_SEARCH_QUERIES     = 3    # total queries per question
NUM_ARTICLES_PER_QUERY = 10   # max articles to fetch per query
RELEVANCE_THRESHOLD    = 4    # 1-6 scale; keep articles rated >= this
TOP_K_ARTICLES         = 15   # keep top K after ranking
NUM_REASONING_PROMPTS  = 3    # number of scratchpad prompts to run

TRACKED_SOURCES = [
    "The New York Times",
    "The Washington Post",
]

# ── Paths ────────────────────────────────────────────────────
REPO_ROOT    = os.path.abspath("../../llm_forecasting")   # adjust if needed
DATA_PATH    = "../../data/validation.json"               # adjust if needed
OUTPUT_DIR   = "results_03"

QUERIES_FILE     = os.path.join(OUTPUT_DIR, "queries.json")
ARTICLES_FILE    = os.path.join(OUTPUT_DIR, "articles.json")
SUMMARIES_FILE   = os.path.join(OUTPUT_DIR, "summaries.json")
PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, "predictions.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

print("Config OK")
print(f"  Model:      {DEFAULT_MODEL}")
print(f"  Questions:  {NUM_QUESTIONS}")
print(f"  Data:       {DATA_PATH}")
print(f"  Output dir: {OUTPUT_DIR}")

# ## 1. Imports and helpers

# In[ ]:


import json
import math
import time
import asyncio
import logging
import os
import sys
import torch
from datetime import datetime, timedelta
from transformers import pipeline as hf_pipeline

import numpy as np
import requests
from bs4 import BeautifulSoup
from gnews import GNews
from googlenewsdecoder import gnewsdecoder

# ── Groq client (via openai-compatible API) ──────────────────
import openai
from config.keys import (
    GROQ_API_KEY_1,
    GROQ_API_KEY_2,
    GROQ_API_KEY_3,
    GROQ_API_KEY_4,
    GROQ_API_KEY_5,
)

GROQ_API_KEYS = [
    GROQ_API_KEY_1,
    GROQ_API_KEY_2,
    GROQ_API_KEY_3,
    GROQ_API_KEY_4,
    GROQ_API_KEY_5,
]

groq_clients = [
    openai.OpenAI(
        api_key=key,
        base_url="https://api.groq.com/openai/v1",
    )
    for key in GROQ_API_KEYS
]

exhausted_keys = set()
current_client_idx = 0

# ── Local LLM setup ──────────────────────────────────────────
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

LOCAL_MODEL_PATH = "/SWS/llms/nobackup/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

local_pipe = None  # loaded lazily on first call

def _get_local_pipe():
    global local_pipe
    if local_pipe is None:
        print("Loading local model (first call only, takes ~2 min)...")
        local_pipe = hf_pipeline(
            "text-generation",
            model=LOCAL_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("Local model ready.")
    return local_pipe

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Groq LLM call ────────────────────────────────────────────
def call_llm_groq(prompt, model=None, temperature=0.0, max_tokens=4096):
    global current_client_idx, exhausted_keys
    model = model or DEFAULT_MODEL
    while True:
        if len(exhausted_keys) == len(groq_clients):
            raise RuntimeError("All Groq API keys are exhausted.")
        if current_client_idx in exhausted_keys:
            current_client_idx = (current_client_idx + 1) % len(groq_clients)
            continue
        client = groq_clients[current_client_idx]
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            err = str(e).lower()
            quota_error = any(x in err for x in [
                "quota", "credit", "exceeded", "rate limit",
                "insufficient", "billing", "429",
            ])
            if quota_error:
                logger.warning(f"API key #{current_client_idx + 1} exhausted. Switching to next key.")
                exhausted_keys.add(current_client_idx)
                current_client_idx = (current_client_idx + 1) % len(groq_clients)
                continue
            logger.warning(f"LLM call failed with key #{current_client_idx + 1}: {e}. Retrying in 10s...")
            time.sleep(10)

# ── Local LLM call ───────────────────────────────────────────
def call_llm_local(prompt, model=None, temperature=0.0, max_tokens=4096):
    pipe = _get_local_pipe()
    # temperature=0 means greedy decoding
    do_sample = temperature > 0
    messages = [{"role": "user", "content": prompt}]
    out = pipe(
        messages,
        max_new_tokens=max_tokens,
        temperature=temperature if do_sample else None,
        do_sample=do_sample,
    )
    return out[0]["generated_text"][-1]["content"]

# ── Unified call_llm — respects USE_LOCAL_LLM flag ───────────
def call_llm(prompt, model=None, temperature=0.0, max_tokens=4096):
    if USE_LOCAL_LLM:
        return call_llm_local(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    else:
        return call_llm_groq(prompt, model=model, temperature=temperature, max_tokens=max_tokens)

# ── JSON file helpers ─────────────────────────────────────────
def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

# ── Date helpers ─────────────────────────────────────────────
def get_single_retrieval_date(date_begin, date_close, date_resolve):
    date_begin_obj   = datetime.strptime(date_begin, "%Y-%m-%d")
    date_close_obj   = datetime.strptime(date_close, "%Y-%m-%d")
    date_resolve_obj = datetime.strptime(date_resolve, "%Y-%m-%d")
    end_date = min(date_close_obj, date_resolve_obj)
    total_days = (end_date - date_begin_obj).days
    if total_days <= 0:
        return date_begin
    offset = min(RETRIEVAL_DAYS_BEFORE_RESOLUTION, total_days)
    retrieval_date = end_date - timedelta(days=offset)
    if retrieval_date < date_begin_obj:
        retrieval_date = date_begin_obj
    return retrieval_date.strftime("%Y-%m-%d")

def get_crowd_prediction_at_date(retrieval_date, community_predictions):
    ref = datetime.strptime(retrieval_date, "%Y-%m-%d")
    candidates = [p for p in community_predictions if datetime.strptime(p[0], "%Y-%m-%d") <= ref]
    if not candidates:
        candidates = community_predictions
    closest = min(candidates, key=lambda p: abs((datetime.strptime(p[0], "%Y-%m-%d") - ref).days))
    return closest[1] if closest else None

def brier_score(prediction, resolution):
    return (prediction - resolution) ** 2

def extract_prediction(response_text):
    import re
    if not response_text:
        return None
    matches = re.findall(r"\*([01]?\.\d+)\*", response_text)
    if matches:
        return float(matches[-1])
    for line in reversed(response_text.strip().split("\n")[-5:]):
        nums = re.findall(r"\b([01]?\.\d+)\b", line)
        for n in reversed(nums):
            v = float(n)
            if 0.0 <= v <= 1.0:
                return v
    return None

print("Cell 1 OK — LLM client and helpers ready")
print(f"  Backend: {'LOCAL (Qwen2.5-7B)' if USE_LOCAL_LLM else 'Groq API'}")

# ## 2. Loading questions

# In[16]:


# ============================================================
# CELL 2 — Load questions from validation.json
# ============================================================

with open(DATA_PATH) as f:
    all_questions = json.load(f)

questions_to_run = all_questions[:NUM_QUESTIONS]

print(f"Loaded {len(all_questions)} questions total.")
print(f"Running on {len(questions_to_run)} question(s):\n")
for i, q in enumerate(questions_to_run):
    community_preds = json.loads(q["community_predictions"])
    retrieval_date = get_single_retrieval_date(
        q["date_begin"],
        q["date_close"],
        q["date_resolve_at"],
    )
    print(f"  [{i}] {q['question']}")
    print(f"       Source:     {q['data_source']}")
    print(f"       Date range: {q['date_begin']} → {q['date_close']}")
    print(f"       Resolution: {q['resolution']} (resolved: {q['is_resolved']})")
    print(f"       Retrieval date:  {retrieval_date}")
    print(f"       Community predictions: {len(community_preds)} data points")
    print()

# ## 3. Generating search queries

# In[17]:


# ============================================================
# CELL 3 — Generate search queries
# Saves to queries.json. Skips questions already processed
# unless FORCE_RERUN_QUERIES = True.
# ============================================================

import re
from prompts.search_query import SEARCH_QUERY_PROMPT_0

FORCE_RERUN_QUERIES = False
MAX_WORDS           = 7
NUM_KEYWORDS        = 3 

def fill_search_query_prompt(template, question, background, date_begin, date_end,
                              num_keywords, max_words):
    prompt_str, _ = template
    return prompt_str.format(
        question=question,
        background=background,
        date_begin=date_begin,
        date_end=date_end,
        num_keywords=num_keywords,
        max_words=max_words,
    )

def extract_queries_from_response(response_text):
    """
    Parse the semicolon-separated queries from the 'Search Queries:' section.
    Returns a list of query strings.
    """
    if not response_text:
        return []
    # Find everything after "Search Queries:"
    match = re.search(r"Search Queries:\s*(.+)", response_text, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    raw = match.group(1).strip()
    # Split on semicolons, clean up
    queries = [q.strip().strip("{}").strip() for q in raw.split(";")]
    queries = [q for q in queries if q and len(q) > 3]
    return queries

# ── Load existing data ────────────────────────────────────────
queries_data   = load_json(QUERIES_FILE)
queries_lookup = {entry["question"]: entry for entry in queries_data}

# ── Main loop ─────────────────────────────────────────────────
for q in questions_to_run:
    question   = q["question"]
    background = q["background"]
    date_close = q["date_close"]
    print(f"\nQuestion: {question[:80]}")

    retrieval_date = get_single_retrieval_date(
        q["date_begin"],
        q["date_close"],
        q["date_resolve_at"],
    )

    # Get or create entry for this question
    if question not in queries_lookup:
        queries_lookup[question] = {
            "question":               question,
            "resolution":             float(q["resolution"]),
            "date_begin":             q["date_begin"],
            "date_close":             q["date_close"],
            "date_resolve_at":        q["date_resolve_at"],
            "retrieval_dates":        [retrieval_date],
            "retrieval_date_queries": {}
        }

    entry           = queries_lookup[question]
    existing_dates  = entry["retrieval_date_queries"]

    for retrieval_date in [retrieval_date]:
        if not FORCE_RERUN_QUERIES and retrieval_date in existing_dates:
            print(f"  [{retrieval_date}] Already generated — skipping.")
            continue

        print(f"  [{retrieval_date}] Generating queries...")
        date_results = {}

        prompt_template = SEARCH_QUERY_PROMPT_0
        prompt_name     = "prompt_0"
        prompt = fill_search_query_prompt(
            prompt_template,
            question=question,
            background=background,
            date_begin=q["date_begin"],
            date_end=retrieval_date,
            num_keywords=NUM_KEYWORDS,
            max_words=MAX_WORDS,
        )
        response = call_llm(
            prompt,
            model=SEARCH_QUERY_MODEL,
            temperature=SEARCH_QUERY_TEMPERATURE,
        )
        queries = extract_queries_from_response(response)

        date_results[prompt_name] = {
            "llm_response": response,
            "queries":      queries,
        }
        print(f"    [{prompt_name}] {len(queries)} queries: {queries}")

        existing_dates[retrieval_date] = date_results

        # Save after every retrieval date
        save_json(QUERIES_FILE, list(queries_lookup.values()))

print(f"\nDone. Saved to {QUERIES_FILE}")

# ## 4. Fetching articles

# In[18]:


# ============================================================
# CELL 4 — Fetch articles from GNews
# For each query:
#   - fetch top articles from GNews
#   - decode URLs
#
# Does NOT scrape full text yet.
#
# Saves to articles.json.
# Skips already-fetched queries unless FORCE_RERUN_ARTICLES = True.
# ============================================================

from gnews import GNews
from googlenewsdecoder import gnewsdecoder
from functools import lru_cache

FORCE_RERUN_ARTICLES = False

def fetch_gnews_articles(query, date_begin, date_end, max_results=10):
    start = datetime.strptime(date_begin, "%Y-%m-%d")
    end   = datetime.strptime(date_end, "%Y-%m-%d")

    gn = GNews(
        language="en",
        country="US",
        start_date=(start.year, start.month, start.day),
        end_date=(end.year, end.month, end.day),
        max_results=max_results,
    )

    try:
        results = gn.get_news(query)
        return results if results else []
    except Exception as e:
        logger.warning(f"GNews fetch failed for '{query}': {e}")
        return []


@lru_cache(maxsize=1000)
def decode_google_news_url(google_url):
    try:
        result = gnewsdecoder(google_url, interval=1)
        if result.get("status"):
            return result["decoded_url"]
    except Exception:
        pass
    return google_url


def parse_article_metadata(raw, query):
    google_url  = raw.get("url", "")
    decoded_url = decode_google_news_url(google_url)

    return {
        "title":          raw.get("title"),
        "published_date": raw.get("published date"),
        "source":         raw.get("publisher", {}).get("title"),
        "source_url":     raw.get("publisher", {}).get("href"),
        "google_news_url": google_url,
        "url":            decoded_url,
        "query":          query,
        "text":           None,   # filled in scraping step
    }


# ── Load data ────────────────────────────────────────────────
queries_data    = load_json(QUERIES_FILE)
articles_data   = load_json(ARTICLES_FILE)
articles_lookup = {e["question"]: e for e in articles_data}


# ── Main loop ────────────────────────────────────────────────
for q_entry in queries_data:
    question = q_entry["question"]
    print(f"\nQuestion: {question[:80]}")

    if question not in articles_lookup:
        articles_lookup[question] = {
            "question": question,
            "resolution": q_entry["resolution"],
            "date_begin": q_entry["date_begin"],
            "date_close": q_entry["date_close"],
            "date_resolve_at": q_entry["date_resolve_at"],
            "retrieval_dates": q_entry["retrieval_dates"],
            "retrieval_date_articles": {}
        }

    entry   = articles_lookup[question]
    date_art = entry["retrieval_date_articles"]
    
    # Track changes for batch saving
    batch_counter = 0
    BATCH_SAVE_SIZE = 10

    for retrieval_date, prompt_results in q_entry["retrieval_date_queries"].items():
        if retrieval_date not in date_art:
            date_art[retrieval_date] = {}

        # Use set for O(1) deduplication instead of list
        all_queries = set()
        for _, prompt_data in prompt_results.items():
            all_queries.update(prompt_data.get("queries", []))

        for query in all_queries:
            if not FORCE_RERUN_ARTICLES and query in date_art[retrieval_date]:
                print(f"  [{retrieval_date}] '{query[:50]}' already exists")
                continue

            print(f"  [{retrieval_date}] Fetching: '{query[:60]}'")

            raw_articles = fetch_gnews_articles(
                query,
                date_begin=q_entry["date_begin"],
                date_end=retrieval_date,
                max_results=10
            )

            print(f"    Returned {len(raw_articles)} articles")

            parsed = []
            for raw in raw_articles:
                meta = parse_article_metadata(raw, query)
                print(f"    → [{meta['source']}] {meta['title'][:60] if meta['title'] else 'N/A'}")
                parsed.append(meta)

            # FLAT LIST (no grouping by source)
            date_art[retrieval_date][query] = parsed
            
            batch_counter += 1
            
            # Save in batches to reduce I/O
            if batch_counter >= BATCH_SAVE_SIZE:
                save_json(ARTICLES_FILE, list(articles_lookup.values()))
                batch_counter = 0
                print(f"    [Saved checkpoint]")

    # Final save for this question's remaining items
    if batch_counter > 0:
        save_json(ARTICLES_FILE, list(articles_lookup.values()))

print("\nDone. Saved to", ARTICLES_FILE)

# ## 5. Scraping articles' full text

# In[20]:


# ============================================================
# CELL 5 — Scrape full text of articles
# Reads articles.json, scrapes full text for each unique URL,
# saves to scraped_articles.json.
# ============================================================

import requests
from bs4 import BeautifulSoup

FORCE_RERUN_SCRAPING = False

SCRAPED_FILE      = os.path.join(OUTPUT_DIR, "scraped_articles.json")
SCRAPE_TIMEOUT    = 1
SCRAPE_MIN_LENGTH = 200

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def scrape_article_text(url):
    if not url or url.startswith("https://news.google.com"):
        return None

    try:
        resp = requests.get(url, headers=HEADERS, timeout=SCRAPE_TIMEOUT)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "form", "noscript", "iframe"]):
            tag.decompose()

        article_tag = (
            soup.find("article") or
            soup.find("main") or
            soup.find("div", {"class": lambda c: c and "article" in c.lower()}) or
            soup.find("div", {"class": lambda c: c and "content" in c.lower()}) or
            soup.body
        )

        if not article_tag:
            return None

        paragraphs = article_tag.find_all("p")
        text = " ".join(
            p.get_text(separator=" ", strip=True) for p in paragraphs
        )
        text = " ".join(text.split())

        if len(text) < SCRAPE_MIN_LENGTH:
            return None

        return text

    except Exception:
        return None


# ── Load data ────────────────────────────────────────────────
articles_data  = load_json(ARTICLES_FILE)
scraped_data   = load_json(SCRAPED_FILE)
scraped_lookup = {e["question"]: e for e in scraped_data}

# ── URL cache (avoid re-scraping globally) ────────────────────
url_cache = {}

for entry in scraped_data:
    for date_articles in entry.get("retrieval_date_articles", {}).values():
        for query_articles in date_articles.values():

            # query_articles should be a list of article dicts
            if isinstance(query_articles, list):
                for art in query_articles:
                    if isinstance(art, dict):
                        url = art.get("url")
                        text = art.get("text")
                        if url and text:
                            url_cache[url] = text

            # defensive fallback (in case of corrupted structure)
            elif isinstance(query_articles, dict):
                for art in query_articles.values():
                    if isinstance(art, list):
                        for a in art:
                            if isinstance(a, dict):
                                url = a.get("url")
                                text = a.get("text")
                                if url and text:
                                    url_cache[url] = text


# ── Main loop ────────────────────────────────────────────────
for art_entry in articles_data:
    question = art_entry["question"]

    print(f"\n{'='*60}")
    print(f"Question: {question[:80]}")
    print('='*60)

    if question not in scraped_lookup:
        scraped_lookup[question] = {
            "question": question,
            "resolution": art_entry["resolution"],
            "date_begin": art_entry["date_begin"],
            "date_close": art_entry["date_close"],
            "date_resolve_at": art_entry["date_resolve_at"],
            "retrieval_dates": art_entry["retrieval_dates"],
            "retrieval_date_articles": {}
        }

    scraped_entry = scraped_lookup[question]
    date_out = scraped_entry["retrieval_date_articles"]

    for retrieval_date, query_articles in art_entry["retrieval_date_articles"].items():
        if retrieval_date not in date_out:
            date_out[retrieval_date] = {}

        print(f"\n  [{retrieval_date}]")

        for query, articles in query_articles.items():
            if not FORCE_RERUN_SCRAPING and query in date_out[retrieval_date]:
                print(f"    '{query[:50]}' already scraped")
                continue

            print(f"    Scraping: '{query[:60]}' ({len(articles)})")

            scraped_articles = []

            for i, art in enumerate(articles):
                url = art.get("url")
                title = (art.get("title") or "")[:60]

                if not FORCE_RERUN_SCRAPING and url in url_cache:
                    text = url_cache[url]
                    status = "cached"
                else:
                    text = scrape_article_text(url)
                    url_cache[url] = text
                    status = "scraped" if text else "failed"
                    time.sleep(0.5)

                print(f"      [{i+1}/{len(articles)}] [{status}] {title}")

                scraped_articles.append({**art, "text": text})

            date_out[retrieval_date][query] = scraped_articles

            save_json(SCRAPED_FILE, list(scraped_lookup.values()))

print("\nDone. Saved to", SCRAPED_FILE)

# ## 6. Ranking relevance

# In[22]:


# ============================================================
# CELL 6 — Relevance ranking
# Reads scraped_articles.json, asks the LLM to rate each
# article's relevance (1-6), keeps those >= RELEVANCE_THRESHOLD,
# saves top K per retrieval date to ranked_articles.json.
# Skips already-ranked queries unless FORCE_RERUN_RANKING = True.
# ============================================================

import re
from prompts.relevance import RELEVANCE_PROMPT_0

FORCE_RERUN_RANKING = False
RANKED_FILE = os.path.join(OUTPUT_DIR, "ranked_articles.json")

q_meta = {q["question"]: q for q in questions_to_run}


def fill_relevance_prompt(article_text, question, background, resolution_criteria):
    prompt_str, _ = RELEVANCE_PROMPT_0
    return prompt_str.format(
        question=question,
        background=background,
        resolution_criteria=resolution_criteria,
        article=article_text,
    )


def extract_relevance_rating(response_text):
    if not response_text:
        return None

    match = re.search(r"Rating:\s*([1-6])", response_text)
    if match:
        return int(match.group(1))

    lines = response_text.strip().split("\n")
    for line in reversed(lines[-5:]):
        m = re.search(r"\b([1-6])\b", line)
        if m:
            return int(m.group(1))

    return None


def build_article_text_for_ranking(art):
    parts = []
    if art.get("title"):
        parts.append(f"Title: {art['title']}")
    if art.get("text"):
        parts.append(art["text"][:2000])
    return "\n".join(parts) if parts else None


# ── Load data ─────────────────────────────────────────────────
scraped_data = load_json(SCRAPED_FILE)
ranked_data = load_json(RANKED_FILE)
ranked_lookup = {e["question"]: e for e in ranked_data}


# ── Main loop ─────────────────────────────────────────────────
for scraped_entry in scraped_data:
    question = scraped_entry["question"]
    q_info = q_meta.get(question, {})

    print(f"\n{'='*60}")
    print(f"Question: {question[:80]}")
    print("="*60)

    if question not in ranked_lookup:
        ranked_lookup[question] = {
            "question": question,
            "resolution": scraped_entry["resolution"],
            "date_begin": scraped_entry["date_begin"],
            "date_close": scraped_entry["date_close"],
            "date_resolve_at": scraped_entry["date_resolve_at"],
            "retrieval_dates": scraped_entry["retrieval_dates"],
            "retrieval_date_rankings": {}
        }

    ranked_entry = ranked_lookup[question]
    date_rankings = ranked_entry["retrieval_date_rankings"]

    for retrieval_date, query_articles in scraped_entry["retrieval_date_articles"].items():

        if not FORCE_RERUN_RANKING and retrieval_date in date_rankings:
            n = date_rankings[retrieval_date]["num_articles_ranked"]
            print(f"\n  [{retrieval_date}] Already ranked ({n}) — skipping.")
            continue

        print(f"\n  [{retrieval_date}] Ranking articles...")

        # ── FLATTEN ALL ARTICLES ACROSS QUERIES ─────────────────
        seen_urls = set()
        pooled = []

        for query, articles in query_articles.items():
            for art in articles:
                url = art.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    pooled.append(art)

        print(f"    Pooled {len(pooled)} unique articles.")

        if not pooled:
            date_rankings[retrieval_date] = {
                "num_articles_pooled": 0,
                "num_articles_rated": 0,
                "num_articles_ranked": 0,
                "all_rated_articles": [],
                "ranked_articles": [],
            }
            save_json(RANKED_FILE, list(ranked_lookup.values()))
            continue

        # ── RATE ONLY ARTICLES WITH FULL TEXT ───────────────────
        all_rated = []

        for i, art in enumerate(pooled):
            title = (art.get("title") or "")[:60]

            # CRITICAL CHANGE: skip missing text entirely
            if not art.get("text"):
                print(f"    [{i+1}/{len(pooled)}] ✗ No text — skipped: {title}")
                continue

            art_text = build_article_text_for_ranking(art)

            if not art_text:
                print(f"    [{i+1}/{len(pooled)}] ✗ Empty text — skipped: {title}")
                continue

            prompt = fill_relevance_prompt(
                article_text=art_text,
                question=question,
                background=q_info.get("background", ""),
                resolution_criteria=q_info.get("resolution_criteria", ""),
            )

            response = call_llm(
                prompt,
                model=RANKING_MODEL,
                temperature=RANKING_TEMPERATURE,
            )

            rating = extract_relevance_rating(response)
            print(f"    [{i+1}/{len(pooled)}] ★{rating} {title}")

            all_rated.append({
                **art,
                "relevance_rating": rating,
                "relevance_response": response,
            })

        # ── FILTER + RANK ───────────────────────────────────────
        ranked = [
            a for a in all_rated
            if a["relevance_rating"] is not None
            and a["relevance_rating"] >= RELEVANCE_THRESHOLD
        ]

        ranked = sorted(ranked, key=lambda x: x["relevance_rating"], reverse=True)
        ranked = ranked[:TOP_K_ARTICLES]

        print(f"    {len(all_rated)} rated → {len(ranked)} kept")

        date_rankings[retrieval_date] = {
            "num_articles_pooled": len(pooled),
            "num_articles_rated": len(all_rated),
            "num_articles_ranked": len(ranked),
            "all_rated_articles": all_rated,
            "ranked_articles": ranked,
        }

        save_json(RANKED_FILE, list(ranked_lookup.values()))

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

for entry in ranked_lookup.values():
    print(f"\n{entry['question'][:70]}")
    for date, dr in entry["retrieval_date_rankings"].items():
        print(
            f"  [{date}] pooled={dr['num_articles_pooled']} "
            f"rated={dr['num_articles_rated']} "
            f"kept={dr['num_articles_ranked']}"
        )

print("\nDone. Saved to", RANKED_FILE)

# ## 7. Summarising

# In[23]:


# ============================================================
# CELL 7 — Summarise ranked articles (SOURCE-AWARE)
# Now also builds combined summaries per source for Cell 8
# ============================================================

from prompts.summarization import SUMMARIZATION_PROMPT_0

FORCE_RERUN_SUMMARIES = False
SUMMARIES_FILE        = os.path.join(OUTPUT_DIR, "summaries.json")

def fill_summarization_prompt(article_text, question, background):
    prompt_str, _ = SUMMARIZATION_PROMPT_0
    return prompt_str.format(
        question=question,
        background=background,
        article=article_text,
    )

# ── Load data ─────────────────────────────────────────────────
ranked_data      = load_json(RANKED_FILE)
summaries_data   = load_json(SUMMARIES_FILE)
summaries_lookup = {entry["question"]: entry for entry in summaries_data}

summary_cache = {}

q_meta = {q["question"]: q for q in questions_to_run}

# ── Main loop ─────────────────────────────────────────────────
for ranked_entry in ranked_data:
    question = ranked_entry["question"]
    q_info   = q_meta.get(question, {})

    print(f"\n{'='*60}")
    print(f"Question: {question[:80]}")
    print('='*60)

    if question not in summaries_lookup:
        summaries_lookup[question] = {
            "question": question,
            "resolution": ranked_entry["resolution"],
            "date_begin": ranked_entry["date_begin"],
            "date_close": ranked_entry["date_close"],
            "date_resolve_at": ranked_entry["date_resolve_at"],
            "retrieval_dates": ranked_entry["retrieval_dates"],
            "retrieval_date_summaries": {}
        }

    sum_entry = summaries_lookup[question]
    date_sums = sum_entry["retrieval_date_summaries"]

    for retrieval_date, date_data in ranked_entry["retrieval_date_rankings"].items():
        if not FORCE_RERUN_SUMMARIES and retrieval_date in date_sums:
            continue

        ranked_articles = date_data.get("ranked_articles", [])
        print(f"\n  [{retrieval_date}] {len(ranked_articles)} articles")

        if not ranked_articles:
            date_sums[retrieval_date] = {
                "num_ranked": 0,
                "num_summarised": 0,
                "summarised_articles": [],
                "combined_summary_by_source": {}
            }
            continue

        summarised = []

        for i, art in enumerate(ranked_articles):
            url   = art.get("url", "")
            title = art.get("title") or "N/A"
            text  = art.get("text")

            if not text:
                summarised.append({**art, "summary": None})
                continue

            if not FORCE_RERUN_SUMMARIES and url in summary_cache:
                summarised.append({**art, "summary": summary_cache[url]})
                continue

            prompt = fill_summarization_prompt(
                article_text=text[:4000],
                question=question,
                background=q_info.get("background", ""),
            )

            response = call_llm(
                prompt,
                model=SUMMARIZATION_MODEL,
                temperature=SUMMARIZATION_TEMPERATURE,
                max_tokens=512,
            )

            summary = response.strip() if response else None
            summary_cache[url] = summary

            summarised.append({
                **art,
                "summary": summary,
                "summary_llm_response": response
            })

        # ── GROUP BY SOURCE ─────────────────────────────
        combined_by_source = {}

        for art in summarised:
            src = art.get("source") or "unknown"

            if not art.get("summary"):
                continue

            block = (
                f"Title: {art.get('title','N/A')}\n"
                f"Publisher: {src}\n"
                f"Published: {art.get('published_date','N/A')}\n"
                f"Summary: {art['summary']}"
            )

            combined_by_source.setdefault(src, []).append(block)

        # join blocks
        for src in combined_by_source:
            combined_by_source[src] = "\n\n".join(combined_by_source[src])

        date_sums[retrieval_date] = {
            "num_ranked": len(ranked_articles),
            "num_summarised": len([a for a in summarised if a.get("summary")]),
            "summarised_articles": summarised,
            "combined_summary_by_source": combined_by_source
        }

        save_json(SUMMARIES_FILE, list(summaries_lookup.values()))

print("Done.")

# ## 8. Getting probabilities

# In[ ]:


# ============================================================
# CELL 8 — SOURCE-LEVEL FORECASTING (simplified)
# Uses precomputed source_combined_summaries from Cell 7
# ============================================================

import re
from prompts.base_reasoning import BINARY_SCRATCH_PAD_PROMPT_0

FORCE_RERUN_PREDICTIONS = False
PREDICTIONS_FILE        = os.path.join(OUTPUT_DIR, "predictions.json")

def fill_reasoning_prompt(question, background, resolution_criteria,
                          date_begin, date_end, retrieved_info):
    prompt_str, _ = BINARY_SCRATCH_PAD_PROMPT_0
    return prompt_str.format(
        question=question,
        background=background,
        resolution_criteria=resolution_criteria,
        date_begin=date_begin,
        date_end=date_end,
        retrieved_info=retrieved_info,
    )

summaries_data     = load_json(SUMMARIES_FILE)
predictions_data   = load_json(PREDICTIONS_FILE)
predictions_lookup = {e["question"]: e for e in predictions_data}

q_meta = {q["question"]: q for q in questions_to_run}

for sum_entry in summaries_data:

    question = sum_entry["question"]
    q_info   = q_meta.get(question, {})

    print(f"\n{'='*60}")
    print(f"Question: {question[:80]}")
    print('='*60)

    if question not in predictions_lookup:
        predictions_lookup[question] = {
            "question": question,
            "resolution": sum_entry["resolution"],
            "date_begin": sum_entry["date_begin"],
            "date_close": sum_entry["date_close"],
            "date_resolve_at": sum_entry["date_resolve_at"],
            "retrieval_date_predictions": {}
        }

    pred_entry = predictions_lookup[question]
    date_preds = pred_entry["retrieval_date_predictions"]

    for retrieval_date, date_data in sum_entry["retrieval_date_summaries"].items():

        if not FORCE_RERUN_PREDICTIONS and retrieval_date in date_preds:
            continue

        combined_by_source = date_data.get("combined_summary_by_source", {})

        if not combined_by_source:
            date_preds[retrieval_date] = {
                "source_predictions": {}
            }
            continue

        source_predictions = {}

        for source, summary_block in combined_by_source.items():

            prompt = fill_reasoning_prompt(
                question=question,
                background=q_info.get("background", ""),
                resolution_criteria=q_info.get("resolution_criteria", ""),
                date_begin=sum_entry["date_begin"],
                date_end=sum_entry["date_close"],
                retrieved_info=summary_block,
            )

            response = call_llm(
                prompt,
                model=REASONING_MODEL,
                temperature=REASONING_TEMPERATURE,
                max_tokens=2048,
            )

            prediction = extract_prediction(response)

            source_predictions[source] = {
                "llm_response": response,
                "prediction": prediction
            }

            print(f"  [{retrieval_date}] {source}: {prediction}")

        date_preds[retrieval_date] = {
            "source_predictions": source_predictions
        }

        save_json(PREDICTIONS_FILE, list(predictions_lookup.values()))

print("Done.")

# ## 9. Analysing results

# In[12]:


# ============================================================
# CELL 9 — Compare Brier scores (Crowd vs Source-level LLM)
# One table per question:
#   - Crowd baseline (fixed cutoff: 7 days pre-resolution)
#   - One column per source
# ============================================================

predictions_data = load_json(PREDICTIONS_FILE)

print(f"\n{'='*120}")
print("SOURCE-LEVEL LLM vs CROWD — BRIER SCORE COMPARISON")
print('='*120)

CUTOFF_DAYS = 7

for entry in predictions_data:

    question   = entry["question"]
    resolution = float(entry["resolution"])

    q_original = next(
        q for q in all_questions
        if q["question"] == question
    )

    community_predictions = json.loads(q_original["community_predictions"])

    # ── Define fixed evaluation date (7 days before resolution)
    eval_date = (
        datetime.strptime(q_original["date_resolve_at"], "%Y-%m-%d")
        - timedelta(days=CUTOFF_DAYS)
    ).strftime("%Y-%m-%d")

    crowd_pred = get_crowd_prediction_at_date(
        eval_date,
        community_predictions
    )

    crowd_brier = (
        brier_score(crowd_pred, resolution)
        if crowd_pred is not None else None
    )

    print(f"\nQuestion: {question}")
    print(f"Evaluation date: {eval_date}")
    print("-" * 120)

    # Collect all sources across all dates
    source_preds = {}

    for retrieval_date, data in entry["retrieval_date_predictions"].items():

        source_block = data.get("source_predictions", {})

        for source, pred_data in source_block.items():

            pred = pred_data.get("prediction")
            if pred is None:
                continue

            source_preds[source] = pred

            pred = pred_data.get("prediction")
            if pred is None:
                continue

            # keep best (or last) prediction per source
            source_preds[source] = pred

    # Build table header
    sources_sorted = sorted(source_preds.keys())

    header = (
        f"{'Crowd':>12}"
        + "".join([f"{s:>20}" for s in sources_sorted])
    )

    print(header)
    print("-" * len(header))

    row = ""

    # Crowd column
    row += f"{round(crowd_brier, 4) if crowd_brier is not None else 'None':>12}"

    # Source columns
    for s in sources_sorted:
        pred = source_preds[s]
        brier = brier_score(pred, resolution)
        row += f"{round(brier, 4):>20}"

    print(row)

print(f"\n{'='*120}")
print("Done.")
print('='*120)

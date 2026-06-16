# Replication

This folder contains a partial replication of [*Approaching Human-Level Forecasting with Language Models*](https://arxiv.org/pdf/2402.18563), adapted to compare LLM forecast quality across different news sources.

The original pipeline uses LLMs to forecast binary prediction market questions by retrieving and summarizing news articles, then producing a probability estimate compared against crowd predictions via Brier score. Our adaptation keeps the same pipeline structure and prompts, but runs the reasoning step separately for each news source — producing one forecast per source per question, allowing direct source-level comparison.

This folder is self-contained. The rest of the repository belongs to the original authors and has its own README.

## What's different from the original paper

- **Models**: the original models are deprecated. We use Llama 3.3 70B via the Groq API, or alternatively Qwen 2.5 (7B or 14B) running on a local compute cluster.
- **Questions**: the paper's questions predate the knowledge cutoff of current LLMs. We use our own Polymarket questions collected in 2026.
- **Article retrieval**: the paper's article dataset is not released and their NewsCatcher setup is not fully documented. We retrieve articles using the GNews Python package only.

## Pipeline

The notebook runs end-to-end in the following steps:

1. **Query generation** — the LLM generates search queries for each market question
2. **Article retrieval** — articles are fetched via GNews and URLs are decoded with `googlenewsdecoder`
3. **Scraping** — full text is scraped from decoded URLs, filtered to a set of tracked sources
4. **Relevance ranking** — the LLM rates each article on a 1–6 scale; articles below the threshold are dropped
5. **Summarization** — top-ranked articles are summarized per source
6. **Forecasting** — the LLM produces a probability estimate per source using only that source's summaries
7. **Evaluation** — Brier scores are computed per source and compared against crowd predictions

## Setup

```bash
pip install -r requirements.txt
```

API keys go in a `.env` file or in `config/keys.py`.

To use a local cluster model instead of Groq, set `USE_LOCAL_LLM = True` in the configuration cell and point `LOCAL_MODEL_PATH` to your model snapshot.

## Configuration

All experiment settings are in the first notebook cell:

| Parameter | Description |
|---|---|
| `NUM_QUESTIONS` | Number of market questions to evaluate |
| `DEFAULT_MODEL` | Groq model to use (ignored if `USE_LOCAL_LLM = True`) |
| `USE_LOCAL_LLM` | Switch between Groq API and local cluster model |
| `RELEVANCE_THRESHOLD` | Minimum relevance score to keep an article (1–6) |
| `TOP_K_ARTICLES` | Max articles to pass to summarization per retrieval date |
| `TRACKED_SOURCES` | List of news sources to include in the comparison |
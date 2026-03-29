from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Agent Factory Lite"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"
    auth_required: bool = False
    default_workspace_name: str = "default"

    # ── LLM provider ─────────────────────────────────────────────────────────
    # "groq" | "anthropic"
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"

    # Groq (preferido — rápido, grátis na tier free, ótimo PT-BR)
    groq_api_key: str = ""

    # Anthropic (fallback / alternativa)
    anthropic_api_key: str = ""

    # ── Embeddings (fastembed — ONNX Runtime, sem PyTorch) ───────────────────
    # sentence-transformers/paraphrase-multilingual-mpnet-base-v2  (768d, geral)
    # intfloat/multilingual-e5-small                               (384d, leve)
    # BAAI/bge-m3                                                  (1024d, alta perf)
    # rufimelo/Legal-BERTimbau-sts-large  ← requer Long Paths habilitado
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_batch_size: int = 32

    # ── Chunking ───────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── FAISS storage path (kept as chroma_path for backward compatibility) ─
    chroma_path: str = "data/chroma"
    # FAISS index type: "flat" (IndexFlatIP) or "hnsw" (IndexHNSWFlat)
    faiss_index_type: str = "flat"
    # HNSW parameters (only used when faiss_index_type="hnsw")
    faiss_hnsw_m: int = 32
    faiss_hnsw_ef_construction: int = 200
    faiss_hnsw_ef_search: int = 64

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = 15
    retrieval_min_score: float = 0.0  # 0.0 = desabilitado; sugerido: 0.01 para distância cosine
    default_domain_profile: str = "general"

    # ── Legal document chunking ────────────────────────────────────────────────
    legal_chunk_max_size: int = 2000
    legal_child_threshold: int = 800
    legal_chunk_overlap: int = 160

    # ── Legal tree + summaries ────────────────────────────────────────────────
    # Build canonical legal tree and index macro chunks (chapters/sections).
    legal_tree_enabled: bool = True
    # Generate pre-computed summaries per chapter/section via LLM.
    # Each chapter/section costs one LLM call — disable to save cost.
    legal_summaries_enabled: bool = True
    # Reuse existing summaries when re-ingesting a document (skip LLM calls
    # if summaries already exist for the same doc_id + collection).
    legal_summaries_cache: bool = True

    # ── Intent classification ─────────────────────────────────────────────────
    # "regex" = fast heuristic patterns (default)
    # "embeddings" = ML-based via embedding similarity (more accurate, slower)
    intent_classifier: str = "regex"

    # ── Intelligent ingestion / tabular ───────────────────────────────────────
    tabular_detection_threshold: float = 0.4
    tabular_chunk_group_size: int = 5
    tabular_chunk_max_chars: int = 512

    # ── Structured store ───────────────────────────────────────────────────────
    structured_store_path: str = "data/structured.duckdb"
    structured_store_enabled: bool = True
    table_planner_llm_enabled: bool = False

    # ── Query routing ──────────────────────────────────────────────────────────
    query_routing_enabled: bool = True

    # ── Structural retrieval tuning ────────────────────────────────────────────
    retrieval_structural_bonus: float = 0.12
    retrieval_adjacency_window: int = 1
    retrieval_section_hint_bonus: float = 0.14

    # ── PDF pipeline ──────────────────────────────────────────────────────────
    # When True, PDFs go through structured block extraction + quality gate.
    pdf_pipeline_enabled: bool = True
    # Minimum quality score (0.0–1.0) to allow automatic indexing.
    pdf_pipeline_min_quality: float = 0.4
    # Save .md and .json artifacts to data/processed/
    pdf_pipeline_save_artifacts: bool = True
    pdf_pipeline_artifacts_dir: str = "data/processed"

    # ── Cross-encoder reranking ────────────────────────────────────────────────
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 10

    # ── Query expansion via LLM ─────────────────────────────────────────────
    query_expansion_enabled: bool = False
    query_expansion_max_reformulations: int = 2

    # ── HyDE (Hypothetical Document Embeddings) ─────────────────────────────
    hyde_enabled: bool = False
    hyde_merge_original: bool = True

    # ── Contextual compression ───────────────────────────────────────────────
    compression_enabled: bool = False
    compression_method: str = "extractive"  # "extractive" | "llm"
    compression_max_sentences: int = 3

    # ── Context budget ─────────────────────────────────────────────────────────
    # Approximate max tokens for context sent to LLM. Chunks are dropped (lowest
    # score first) when the total exceeds this budget.
    max_context_tokens: int = 6000
    # Avg chars per token for budget estimation (conservative for PT-BR)
    chars_per_token: float = 3.5

    # ── Modo Simples ──────────────────────────────────────────────────────────
    # Quando True: UI embutida em /, rotas enterprise escondidas, auth desligado.
    simple_mode: bool = True
    default_collection: str = "documentos"


@lru_cache
def get_settings() -> Settings:
    return Settings()

from __future__ import annotations


class AppConfigMixin:
    app_name: str = "Agent Factory Lite"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"
    auth_required: bool = False
    default_workspace_name: str = "default"
    simple_mode: bool = True
    default_collection: str = "documentos"


class LLMConfigMixin:
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str = ""
    anthropic_api_key: str = ""


class EmbeddingConfigMixin:
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_batch_size: int = 32


class StorageConfigMixin:
    chunk_size: int = 512
    chunk_overlap: int = 64
    chroma_path: str = "data/chroma"
    faiss_index_type: str = "auto"
    faiss_hnsw_m: int = 32
    faiss_hnsw_ef_construction: int = 200
    faiss_hnsw_ef_search: int = 64
    structured_store_path: str = "data/structured.duckdb"
    structured_store_enabled: bool = True


class RetrievalConfigMixin:
    retrieval_top_k: int = 15
    retrieval_min_score: float = 0.0
    default_domain_profile: str = "general"
    retrieval_structural_bonus: float = 0.12
    retrieval_adjacency_window: int = 1
    retrieval_section_hint_bonus: float = 0.14
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 10
    query_expansion_enabled: bool = False
    query_expansion_max_reformulations: int = 2
    hyde_enabled: bool = False
    hyde_merge_original: bool = True
    compression_enabled: bool = False
    compression_method: str = "extractive"
    compression_max_sentences: int = 3
    max_context_tokens: int = 6000
    chars_per_token: float = 3.5


class LegalConfigMixin:
    legal_chunk_max_size: int = 2000
    legal_child_threshold: int = 800
    legal_chunk_overlap: int = 160
    legal_tree_enabled: bool = True
    legal_summaries_enabled: bool = True
    legal_summaries_cache: bool = True
    intent_classifier: str = "regex"


class TabularConfigMixin:
    tabular_detection_threshold: float = 0.4
    tabular_chunk_group_size: int = 5
    tabular_chunk_max_chars: int = 512
    table_planner_llm_enabled: bool = False
    query_routing_enabled: bool = True


class PdfPipelineConfigMixin:
    pdf_pipeline_enabled: bool = True
    pdf_pipeline_min_quality: float = 0.4
    pdf_pipeline_save_artifacts: bool = True
    pdf_pipeline_artifacts_dir: str = "data/processed"

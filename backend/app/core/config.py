from functools import lru_cache
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str
    frontend_url: str = "http://localhost:5173"
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    rag_chunk_max_tokens: int = 480
    rag_chunk_tokenizer: str = "bert-base-multilingual-cased"
    rag_debug_export_chunks: bool = False
    rag_extraction_min_quality_score: float = 0.65
    rag_native_high_confidence_score: float = 0.85
    rag_native_borderline_score: float = 0.60
    rag_ocr_min_selection_gain: float = 0.03
    rag_ocr_enabled: bool = True
    rag_ocr_engine: str = "easyocr"
    rag_ocr_languages: str = "ar,en"
    rag_ocr_page_timeout_seconds: float = 180.0
    rag_model_download_max_retries: int = 2
    rag_model_download_retry_seconds: float = 0.25
    rag_full_ocr_page_ratio: float = 0.60
    rag_extraction_cache_enabled: bool = True
    rag_extraction_pipeline_version: str = "2026-08-16.2"
    rag_preflight_analysis_version: str = "2026-08-18.1"
    rag_preflight_native_only_max_bad_ratio: float = 0.02
    rag_preflight_ocr_heavy_ratio: float = 0.60
    rag_preflight_batch_size: int = 30
    rag_preflight_min_batch_size: int = 5
    rag_allow_partial_ingestion: bool = True
    rag_max_failed_page_ratio: float = 0.20
    rag_min_valid_chunks: int = 1
    rag_embedding_provider: str = "qwen3"
    rag_embedding_model_id: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        validation_alias=AliasChoices("RAG_EMBEDDING_MODEL", "RAG_EMBEDDING_MODEL_ID"),
    )
    rag_embedding_dimension: int = 1024
    rag_embedding_device: str = "auto"
    # Qwen supports much longer contexts, but teacher-library chunks are
    # intentionally small. Bounding the CPU attention window prevents a
    # malformed/oversized chunk from turning one background task into minutes
    # of unobservable work.
    rag_embedding_max_seq_length: int = 1024
    rag_embedding_cpu_threads: int = 8
    rag_embedding_query_instruction: str = (
        "Given a pedagogical query, retrieve relevant passages from multilingual "
        "Arabic-language teaching resources and CEFR-aligned educational documents."
    )
    rag_embedding_config_version: str = "v3"
    rag_embedding_batch_size: int = 8
    rag_reranker_model_id: str = Field(
        default="Qwen/Qwen3-Reranker-0.6B",
        validation_alias=AliasChoices("RAG_RERANKER_MODEL", "RAG_RERANKER_MODEL_ID"),
    )
    rag_reranker_device: str = "auto"
    rag_reranker_batch_size: int = 8
    rag_reranker_instruction: str = (
        "Given a pedagogical query, rank passages by how directly and completely "
        "they answer the query. Prefer passages containing explicit requested "
        "information such as learning objectives, lesson goals, competencies, "
        "instructions, assessment criteria, or pedagogical content over passages "
        "that are only topically related."
    )
    rag_retrieval_candidate_top_k: int = 20
    rag_retrieval_final_top_k: int = 5
    rag_reranker_enabled: bool = False
    rag_retrieval_top_k: int = 10
    retrieval_mode: str = "dense"
    hybrid_dense_top_k: int = 20
    hybrid_sparse_top_k: int = 20
    hybrid_rrf_k: int = 60
    pedagogical_retrieval_pipeline_mode: str = "legacy"
    pedagogical_ranking_enabled: bool = False
    pedagogical_context_composition_enabled: bool = False
    pedagogical_context_composition_pool_size: int = 20
    rag_context_max_chunks: int = 6
    rag_context_max_tokens: int = 1800
    rag_llm_provider: str = "openai_compatible"
    rag_llm_model: str = ""
    rag_llm_base_url: str = ""
    rag_llm_api_key: str | None = None
    rag_llm_timeout_seconds: float = 60.0
    rag_llm_temperature: float = 0.2
    rag_llm_max_tokens: int = 1200
    assistant_llm_max_output_tokens: int = 1800
    # JSON validation drafts need room for delimiters and closed-world facts;
    # keep this independent from answer-oriented RAG completions.
    structured_generation_max_output_tokens: int = 2000
    lesson_plan_max_output_tokens: int = 4096
    rag_llm_max_retries: int = 3
    rag_llm_retry_base_delay: float = 1.0
    rag_llm_retry_max_delay: float = 8.0
    # The reviewer is optional: two total primary attempts avoid amplifying a
    # provider's rate limit before trying the configured fallback.
    arabic_review_max_retries: int = 1
    arabic_review_max_wait_seconds: float = 20.0
    arabic_review_retry_base_delay: float = 2.0
    arabic_review_retry_max_delay: float = 8.0
    arabic_review_retry_jitter_seconds: float = 0.25
    arabic_review_fallback_enabled: bool = False
    arabic_review_fallback_provider: str = "openai_compatible"
    arabic_review_fallback_base_url: str = ""
    arabic_review_fallback_api_key: str | None = None
    arabic_review_fallback_model: str = ""
    arabic_review_fallback_timeout_seconds: float = 30.0
    arabic_review_fallback_max_retries: int = 0
    arabic_review_max_output_tokens: int = 3000
    arabic_review_reasoning_effort: str = "medium"
    arabic_review_include_reasoning: bool = False
    assistant_history_max_messages: int = 8
    assistant_history_max_chars: int = 12000
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "knowledge_chunks"
    qdrant_timeout: float = 30.0
    # H2: an opt-in Qdrant sparse representation. Dense retrieval never reads it
    # until a later hybrid-retrieval phase explicitly opts in.
    rag_sparse_vector_name: str = "lexical_sparse"
    knowledge_job_max_attempts: int = 2
    knowledge_job_stale_minutes: int = 30
    knowledge_worker_poll_seconds: float = 2.0
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

@lru_cache
def get_settings() -> Settings:
    return Settings()

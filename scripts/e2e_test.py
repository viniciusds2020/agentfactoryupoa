"""End-to-end test: PDF pipeline -> ingest -> retrieval -> answer quality."""
from __future__ import annotations

import json
import os
import shutil
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Setup: clean state ──────────────────────────────────────────────────────

TEST_CHROMA = "data/chroma_e2e_test"
TEST_ARTIFACTS = "data/processed_e2e_test"

# Patch settings before any import
os.environ["CHROMA_PATH"] = TEST_CHROMA
os.environ["PDF_PIPELINE_ARTIFACTS_DIR"] = TEST_ARTIFACTS
os.environ["PDF_PIPELINE_ENABLED"] = "true"
os.environ["PDF_PIPELINE_SAVE_ARTIFACTS"] = "true"
os.environ["PDF_PIPELINE_MIN_QUALITY"] = "0.3"

# Clean previous test data
for d in (TEST_CHROMA, TEST_ARTIFACTS):
    if os.path.exists(d):
        shutil.rmtree(d)


def separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ── Step 1: PDF Pipeline ────────────────────────────────────────────────────

separator("STEP 1: PDF PIPELINE")

from src.pdf_pipeline import process_pdf

for pdf_name in ["estatuto_cooperativa.pdf", "politica_rh.pdf"]:
    pdf_path = f"data/docs/{pdf_name}"
    result = process_pdf(
        pdf_path=pdf_path,
        save_artifacts_flag=True,
        artifacts_dir=TEST_ARTIFACTS,
        min_quality_score=0.3,
    )

    print(f"\n--- {pdf_name} ---")
    print(f"  Status:        {result.status}")
    print(f"  Blocks:        {len(result.blocks)}")
    print(f"  Quality score: {result.quality.score:.3f}")
    print(f"  Quality flags: {result.quality.flags or '(none)'}")
    print(f"  Noise ratio:   {result.quality.noise_ratio:.3f}")
    print(f"  Structure:     {result.quality.structure_detected}")
    print(f"  Clean text:    {len(result.clean_text)} chars")

    # Block type distribution
    types = {}
    for b in result.blocks:
        types[b.block_type] = types.get(b.block_type, 0) + 1
    print(f"  Block types:   {dict(sorted(types.items()))}")

    if result.md_path:
        print(f"  MD artifact:   {result.md_path}")
    if result.json_path:
        print(f"  JSON artifact: {result.json_path}")

# ── Step 2: Ingest ──────────────────────────────────────────────────────────

separator("STEP 2: INGESTION")

from src.config import Settings

# Create fresh settings for e2e test
settings = Settings(
    chroma_path=TEST_CHROMA,
    pdf_pipeline_enabled=True,
    pdf_pipeline_save_artifacts=True,
    pdf_pipeline_artifacts_dir=TEST_ARTIFACTS,
    pdf_pipeline_min_quality=0.3,
    embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)

# Monkey-patch get_settings for ingestion
import src.config
src.config.get_settings = lambda: settings

from src.ingestion import ingest

for pdf_name, profile in [("estatuto_cooperativa.pdf", "legal"), ("politica_rh.pdf", "hr")]:
    pdf_path = f"data/docs/{pdf_name}"
    doc_id = pdf_name.replace(".pdf", "")
    try:
        chunks = ingest(
            collection="e2e_test",
            source=pdf_path,
            doc_id=doc_id,
            domain_profile=profile,
        )
        print(f"  {pdf_name}: {chunks} chunks indexed (profile={profile})")
    except Exception as e:
        print(f"  {pdf_name}: FAILED - {e}")

# ── Step 3: Retrieval Quality ────────────────────────────────────────────────

separator("STEP 3: RETRIEVAL QUALITY")

from src.chat import _vector_retrieve
from src import vectordb

physical = vectordb.resolve_query_collection(
    "e2e_test",
    settings.embedding_model,
    workspace_id="default",
)
print(f"  Physical collection: {physical}")

# Test queries with expected content
QUERIES = [
    # Estatuto (legal)
    {
        "question": "Quem pode associar-se a cooperativa?",
        "expected_terms": ["Art. 7", "pessoas fisicas", "18 anos", "requisitos"],
        "category": "legal-admissao",
    },
    {
        "question": "Quem nao pode se associar?",
        "expected_terms": ["Art. 8", "politico-partidario", "impedimento", "emprego"],
        "category": "legal-vedacao",
    },
    {
        "question": "Qual o valor das quotas-partes?",
        "expected_terms": ["R$ 100", "quotas-partes", "minimo", "10"],
        "category": "legal-capital",
    },
    {
        "question": "Como funciona a Assembleia Geral?",
        "expected_terms": ["orgao supremo", "convocada", "10 dias", "edital"],
        "category": "legal-assembleia",
    },
    {
        "question": "Quantos membros tem o Conselho de Administracao?",
        "expected_terms": ["7", "membros", "4", "anos", "mandato"],
        "category": "legal-conselho",
    },
    # Politica RH
    {
        "question": "Qual o valor do vale refeicao?",
        "expected_terms": ["35", "dia util"],
        "category": "rh-beneficios",
    },
    {
        "question": "Quanto tempo dura a licenca-maternidade?",
        "expected_terms": ["180 dias", "Empresa Cidada"],
        "category": "rh-licenca",
    },
    {
        "question": "Qual o prazo para pagar a rescisao?",
        "expected_terms": ["10 dias", "verbas rescisorias"],
        "category": "rh-rescisao",
    },
    {
        "question": "Como funciona o banco de horas?",
        "expected_terms": ["6 meses", "compensado", "horas extras"],
        "category": "rh-jornada",
    },
    {
        "question": "Quais os documentos necessarios para admissao?",
        "expected_terms": ["CTPS", "RG", "CPF", "comprovante"],
        "category": "rh-admissao",
    },
]

total_hits = 0
total_expected = 0

for q in QUERIES:
    results = _vector_retrieve(
        question=q["question"],
        physical_collection=physical,
        top_k=5,
        settings=settings,
    )

    # Check if expected terms appear in top results
    all_text = " ".join(r.get("text", "") for r in results[:3]).lower()
    hits = sum(1 for t in q["expected_terms"] if t.lower() in all_text)
    total = len(q["expected_terms"])
    total_hits += hits
    total_expected += total
    pct = (hits / total * 100) if total else 0

    status = "OK" if pct >= 50 else "WEAK" if pct > 0 else "MISS"
    print(f"  [{status:4s}] {q['category']:20s} | {q['question']}")
    print(f"         terms: {hits}/{total} ({pct:.0f}%) | top-1: {results[0]['text'][:80]}..." if results else "         NO RESULTS")

overall = (total_hits / total_expected * 100) if total_expected else 0
print(f"\n  OVERALL: {total_hits}/{total_expected} terms found ({overall:.1f}%)")

# ── Step 4: Full Answer (mocked LLM) ────────────────────────────────────────

separator("STEP 4: FULL ANSWER PIPELINE (mock LLM)")

from src.chat import answer, ChatResult
import src.llm

# Mock LLM to just echo context size
original_chat = src.llm.chat


def mock_chat(messages, system=""):
    user_msg = messages[-1]["content"] if messages else ""
    ctx_size = len(user_msg)
    # Count [N] references in context
    import re
    refs = re.findall(r"\[(\d+)\]", user_msg)
    max_ref = max(int(r) for r in refs) if refs else 0
    return f"[Mock LLM] Recebi contexto com {ctx_size} chars e {max_ref} fontes. Resposta baseada nas fontes [1]."


src.llm.chat = mock_chat

ANSWER_QUERIES = [
    "Quem pode associar-se a cooperativa?",
    "Qual o prazo para pagar a rescisao?",
    "Quantos membros tem o Conselho Fiscal?",
    "Quais beneficios a empresa oferece?",
]

for q in ANSWER_QUERIES:
    result = answer(
        collection="e2e_test",
        question=q,
        request_id="e2e-test",
    )
    print(f"  Q: {q}")
    print(f"  A: {result.answer[:120]}...")
    print(f"  Sources: {len(result.sources)} | doc_ids: {[s.doc_id for s in result.sources]}")
    print()

src.llm.chat = original_chat

# ── Cleanup ──────────────────────────────────────────────────────────────────

separator("CLEANUP")
for d in (TEST_CHROMA, TEST_ARTIFACTS):
    if os.path.exists(d):
        shutil.rmtree(d)
        print(f"  Removed {d}")

print("\n  E2E test complete.")

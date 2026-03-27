# Agent Factory — Instruções do Projeto

## Objetivo
RAG semântica (vector-only) para documentos corporativos em PT-BR (jurídico, RH).
Arquitetura flat: sem camadas, sem DI framework, fácil de entender e modificar.

## Estrutura
```
app.py              → FastAPI entry point, rotas, middleware ASGI, rate limiting
src/
  __init__.py       → Pacote Python
  config.py         → Settings via pydantic-settings (.env)
  llm.py            → Chamadas LLM (Groq/Anthropic) e embeddings (fastembed/sentence-transformers)
  vectordb.py       → ChromaDB (persistente em data/chroma/)
  lexical.py        → Tokenização e normalização PT-BR (stopwords, stemming RSLP, numerais romanos)
  ingestion.py      → Parse + limpeza estrutural + chunk + embed + upsert + árvore jurídica + macro indexing
  legal_tree.py     → Árvore jurídica canônica (título > capítulo > seção > artigo)
  summaries.py      → Resumos pré-computados por capítulo/seção via LLM
  chat.py           → Roteador de intenção + summary shortcut + retrieval híbrido + reranking + streaming SSE
  reranker.py       → Cross-encoder reranking (sentence-transformers CrossEncoder)
  compressor.py     → Compressão contextual extrativa (reduz ruído nos chunks)
  evaluation.py     → Métricas offline + avaliação de sumarização estrutural
  guardrails.py     → Prompt injection detection, input sanitization, rate limiting
  prompts.py        → Templates PT-BR com citações [N], tom conversacional
  history.py        → Histórico de conversas (SQLite)
  controlplane.py   → Workspaces, documentos, jobs, audit, document_nodes, document_summaries
  observability.py  → Métricas e timers
  utils.py          → Logging JSON, request_id, chunk ID determinístico
frontend/
  src/              → React + Vite + TypeScript
  public/           → Assets estáticos
data/
  docs/             → Documentos brutos enviados pelo usuário
  chroma/           → Índice vetorial persistente
  history.db        → Histórico SQLite
scripts/
  start-backend.bat → Inicia uvicorn na porta 8001
  start-frontend.bat→ Inicia Vite dev server
tests/              → Testes unitários (115+)
```

## Embeddings PT-BR (HuggingFace sentence-transformers)
| Domínio       | Modelo recomendado                                              | Dimensão |
|---------------|------------------------------------------------------------------|----------|
| Geral / RH    | sentence-transformers/paraphrase-multilingual-mpnet-base-v2     | 768      |
| Jurídico      | rufimelo/Legal-BERTimbau-sts-large                              | 1024     |
| Multilingual  | intfloat/multilingual-e5-large                                  | 1024     |

Configurar via `EMBEDDING_MODEL` no `.env`. Dimensão sempre consultada via `llm.embedding_dimension()`.

## Segurança (guardrails)
- **Prompt injection**: 13 padrões regex (PT-BR + EN) bloqueiam manipulação do LLM na pergunta
- **Document injection**: chunks de contexto são sanitizados antes de ir ao LLM (`<system>`, `[INST]`, "ignore instruções" → removidos)
- **Input validation**: question máx 2000 chars; collection só `[a-zA-Z0-9_-]` máx 64 chars; history máx 20 mensagens, roles restritos a user/assistant
- **Rate limiting**: chat 30 req/min por IP, ingest 10 req/min por IP (in-memory, sliding window)
- Novos padrões de ataque devem ser adicionados em `src/guardrails.py`

## Regras obrigatórias
- ChromaDB sempre persistente em `data/chroma/` — nunca EphemeralClient
- Dimensão de embedding nunca hardcodada — consultar modelo em runtime via `llm.embedding_dimension()`
- Query original do usuário sempre preservada (rewriting é aditivo)
- IDs de chunk determinísticos: SHA-256(collection + doc_id + chunk_index)
- Logging sempre JSON estruturado com request_id
- Nunca misturar embeddings de modelos diferentes na mesma collection
- Document parsing: docling (PDF com OCR, DOCX, PPTX), openpyxl (XLSX/XLS), csv (CSV) — nunca unstructured
- Sempre criar testes ao alterar comportamento
- Todo input do usuário passa por guardrails antes de chegar ao LLM

## Qualidade
- Tipagem forte (Python 3.12+)
- Funções pequenas e focadas
- Testes unitários para lógica de chunking, retrieval, prompts e guardrails
- pyproject.toml como fonte de verdade para dependências

## Antes de concluir qualquer tarefa
- rodar testes
- mostrar arquivos alterados
- resumir riscos
- listar próximos passos

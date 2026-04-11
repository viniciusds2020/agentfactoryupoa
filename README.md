# Agent Factory

RAG hibrida otimizada para documentos corporativos em PT-BR. Envie PDF, DOCX, XLSX, CSV ou TXT e receba respostas com citacoes das fontes.

## Inicio Rapido (modo simples)

Requisito: Python 3.12+ instalado.

```bash
# 1. Clone o repositorio
# 2. Execute:
start.bat
```

O script cria o ambiente, instala dependencias, pede sua chave Groq (gratuita em https://console.groq.com/keys), e abre o navegador em `http://localhost:8001`.

Pronto. Envie um documento e faca perguntas.

## Modo Avancado (React frontend + todas as features)

Para usar workspaces, audit trail, metricas, evaluation e o frontend React completo:

1. Defina `SIMPLE_MODE=false` no `.env`
2. Instale as dependencias completas:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cd frontend && npm install
```

3. Inicie os dois processos:

```bash
scripts\start-backend.bat   # porta 8001
scripts\start-frontend.bat  # porta 5173
```

## Configuracao

**Modo simples** â€” apenas 1 variavel no `.env`:

```env
GROQ_API_KEY=gsk_...
```

**Modo avancado** â€” todas as opcoes:

```env
SIMPLE_MODE=false
GROQ_API_KEY=gsk_...
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
ANTHROPIC_API_KEY=
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
RETRIEVAL_TOP_K=15
RERANK_ENABLED=true
CROSS_ENCODER_ENABLED=false
QUERY_EXPANSION_ENABLED=false
AUTH_REQUIRED=false
```

Veja `.env.example` para a lista completa.

### Tuning de Indice FAISS

```env
# flat (IndexFlatIP) -> melhor recall, mais RAM/latencia
# hnsw (IndexHNSWFlat) -> menor latencia em colecoes grandes
FAISS_INDEX_TYPE=flat

# Parametros usados quando FAISS_INDEX_TYPE=hnsw
FAISS_HNSW_M=32
FAISS_HNSW_EF_CONSTRUCTION=200
FAISS_HNSW_EF_SEARCH=64
```

## Funcionalidades

- Ingere PDF, DOCX, PPTX, XLSX, XLS, CSV, TXT e MD
- Modulo dedicado para catalogos e bases tabulares em PDF com indexacao configuravel por colunas esperadas
- Consulta de catalogo por codigo, descricao e filtros simples com fonte e pagina obrigatorias
- Frontend React separado em abas: `Chat` e `Ingestao`
- Pipeline por camadas: Bronze (upload), Prata (extracao markdown/json), Ouro (indexacao vetorial)
- Progresso por estagio em tempo real nos jobs de ingestao (`stage` + `progress_pct`)
- Preview de artefatos Prata (Markdown/JSON) na aba de ingestao
- Download de artefatos Prata (`.md` e `.json`) por documento
- Busca hibrida: vetorial + BM25 + RRF (Reciprocal Rank Fusion)
- Reranking em 2 estagios: deterministico (lexical + metadata) + cross-encoder opcional
- Score final do reranker calibrado (vetorial + cross-encoder normalizado em 0..1)
- Pesos dinamicos BM25/vetor baseados no tipo de query (literal vs conceitual)
- Chunking hierarquico para documentos juridicos (artigo > paragrafo > inciso)
- Parent-child retrieval com expansao de siblings e referencias internas
- Resgate estrutural para queries normativas ("capitulo X", "secao Y", "artigo Z")
- Chunks contextualizados para embedding (metadados hierarquicos no vetor, texto cru armazenado)
- Indexacao hibrida texto+JSON: embeddings recebem contexto estrutural dos blocos PDF (pagina, secoes e tipos de bloco)
- **Arvore juridica canonica**: representacao JSON hierarquica (titulo > capitulo > secao > artigo > paragrafo > inciso)
- **Indice macro por capitulo/secao**: capitulos e secoes indexados como chunks especiais no FAISS
- **Resumos pre-computados por capitulo**: resumo executivo, juridico, pontos-chave, obrigacoes, restricoes e definicoes via LLM
- **Roteador de intencao**: classifica queries em `count_structural`, `list_structural`, `locate_structural`, `contains_structural`, `summary_structural`, `question_structural`, `question_factual`, `locate_excerpt`, `comparison`
- **Structure-first automatico para navegacao documental**: perguntas como "quantos capitulos tem?", "quais artigos estao no capitulo II?" e "o capitulo V tem secao III?" sao respondidas direto pela estrutura da Prata, sem depender do vetor
- **Table-first para planilhas e CSV**: perguntas analiticas como "qual e a renda do estado do Rio de Janeiro?" sao planejadas e executadas no store estruturado antes da RAG textual
- **Catalog-first para codigos e tabelas de referencia**: tabelas de procedimentos, rol, coberturas e regras operacionais passam a responder por lookup de registro, descricao, cobertura, prazo e autorizacao em vez de agregacoes numericas
- **Fontes analiticas explicitas**: quando a resposta vem de consulta tabular, a interface mostra a consulta analitica executada e um resumo do resultado em vez de top 5 trechos
- **Badge visual no chat para catalogo**: respostas vindas do modo `Catalogo/Codigos` exibem um selo explicito (`Resposta por Catalogo/Codigos`)
- **Sugestao automatica de modo antes do upload**: a aba de ingestao analisa nome/extensao do arquivo e sugere `Conversa Geral`, `Juridico/Contratos` ou `Planilhas/Tabelas`; para tabular, tambem sugere subtipo `Catalogo/Codigos` ou `Analitico`
- **Exemplos reais no painel semantico**: a inspecao da base mostra exemplos concretos por coluna a partir do `value_catalog`
- **Shortcut para sumarizacao estrutural**: "resuma o capitulo X" usa resumo pre-computado quando existir; se nao existir, resolve o escopo direto pelo artefato JSON da Prata e gera resumo on-demand
- **Fallback deterministico para summary**: se o LLM falhar ou devolver JSON invalido, o sistema gera um resumo extrativo local (`fallback_only`) em vez de falhar
- **Limpeza estrutural pesada**: merge de linhas quebradas, deduplicacao de paragrafos, remocao de numeros de pagina orfaos
- **Armazenamento semantico dual**: nos estruturais e resumos persistidos em SQLite (document_nodes + document_summaries)
- **Avaliacao de sumarizacao estrutural**: structural_hit@1, article_coverage, summary_found_rate, summary_valid_rate, summary_scope_accuracy, citation_scope_precision, fallback_success_rate
- Fallback automatico LLM: Groq 429 → modelos alternativos → Anthropic
- Streaming SSE de respostas token a token
- Respostas em PT-BR com citacoes das fontes [N]
- Historico de conversas (SQLite)
- Perfis de dominio: geral, juridico, RH
- Guardrails: deteccao de prompt injection (PT-BR + EN), sanitizacao de contexto
- Auto-deteccao de documentos juridicos (sem necessidade de configuracao manual)

## Arquitetura

## Agente de Catalogo Tabular

Este repositorio agora inclui um fluxo dedicado para catalogos e bases tabulares em PDF, pensado para o nosso processo: primeiro estruturar os registros, depois consultar com rastreabilidade, sem depender de RAG semantica complexa.

O perfil medico continua sendo um caso de uso forte, mas o desenho foi feito para reaproveitamento em outros catalogos e tabelas de referencia com estrutura semelhante.

### O que ele faz

- tenta extracao em camadas: `camelot` -> `tabula` -> `PyMuPDF` -> `docling`
- normaliza colunas via configuracao JSON
- indexa os registros em `data/medical_catalog/<catalog_id>/index.json`
- responde apenas com base nos campos extraidos do catalogo ou base tabular
- sempre devolve fonte e pagina de origem
- nao inventa informacoes ausentes
- funciona bem para catálogos de procedimentos, tabelas de codigos, bases de cobertura, regras operacionais e referencias semelhantes
- aceita variacao de layout desde que a estrutura possa ser descrita por colunas esperadas e aliases

### Endpoints dedicados

| Metodo | Rota | Descricao |
|--------|------|-----------|
| `POST` | `/reindex` | reprocessa um PDF de catalogo/base tabular e recria o indice |
| `GET` | `/procedure/{codigo}` | busca exata por codigo |
| `POST` | `/ask` | consulta por pergunta textual + filtros |

### Exemplo rapido

PDF de exemplo: [data/docs/rol_procedimentos_teste.pdf](data/docs/rol_procedimentos_teste.pdf)  
Config de exemplo: [data/catalog_configs/rol_procedimentos_teste.json](data/catalog_configs/rol_procedimentos_teste.json)

```bash
python scripts/ingest_medical_catalog.py ^
  --catalog-id rol_teste ^
  --pdf-path data/docs/rol_procedimentos_teste.pdf ^
  --config-path data/catalog_configs/rol_procedimentos_teste.json
```

```bash
curl -X POST http://localhost:8001/reindex ^
  -H "Content-Type: application/json" ^
  -d "{\"catalog_id\":\"rol_teste\",\"pdf_path\":\"data/docs/rol_procedimentos_teste.pdf\",\"config_path\":\"data/catalog_configs/rol_procedimentos_teste.json\"}"
```

```bash
curl "http://localhost:8001/procedure/20202020?catalog_id=rol_teste"
```

```bash
curl -X POST http://localhost:8001/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"catalog_id\":\"rol_teste\",\"question\":\"Quais procedimentos sao de emergencia?\",\"limit\":3}"
```

### Estrutura nova

- `src/medical_catalog/extraction.py`: extracao com fallback
- `src/medical_catalog/normalize.py`: normalizacao com Pandas
- `src/medical_catalog/service.py`: reindex, lookup e ask
- `src/medical_catalog/api.py`: endpoints FastAPI
- `scripts/ingest_medical_catalog.py`: ingestao via CLI
- `tests/test_medical_catalog_service.py`: testes do fluxo principal
- `tests/test_medical_catalog_routes.py`: testes das rotas dedicadas

### Escopo

Este modulo e generico para catalogos e bases tabulares semelhantes, nao apenas para o `Rol de Procedimentos`.

Cobertura esperada:
- catalogos medicos com codigo, descricao e regras
- tabelas de cobertura, SLA, autorizacao e segmentacao
- bases tabulares com colunas previsiveis e lookup por identificador

Nao e o foco principal:
- PDFs puramente narrativos
- documentos sem estrutura tabular reconhecivel
- layouts totalmente livres sem padrao minimo de colunas

```text
Navegador (UI embutida ou React)
   |
   v
FastAPI (app.py)
   |--- Chat (hibrido + reranking + streaming SSE)
   |    |--- Table-first: agregacao/group-by/ranking em DuckDB/SQLite para planilhas e CSV
   |    |--- Catalog-first: lookup por codigo/descricao/cobertura/autorizacao para tabelas de catalogo
   |    |--- Roteador de intencao (count / list / locate / contains / summary / question / comparison)
   |    |--- Structure-first: contagem, inventario, localizacao e composicao estrutural via artefato JSON da Prata
   |    |--- Shortcut: resumo pre-computado (quando summary_structural)
   |    |--- Fallback estrutural: artefato JSON da Prata -> resolve capitulo/secao/titulo -> summary on-demand
   |    |--- Pipeline padrao: retrieval + reranking + geracao (demais intencoes)
   |
   |--- Ingestion Bronze/Prata/Ouro (jobs + progresso + artefatos)
   |    |--- Limpeza estrutural pesada (merge lines, dedup, normalize)
   |    |--- Arvore juridica canonica (titulo > capitulo > secao > artigo)
   |    |--- Indexacao macro (capitulos/secoes como chunks FAISS)
   |    |--- Resumos pre-computados via LLM (por capitulo/secao)
   |    |--- Fallback extrativo local para garantir summary persistivel mesmo sem JSON valido do LLM
   |
   |--- History (SQLite)
   |
   +--> FAISS (persistente em disco)
   |    |--- Nivel 1: Macro retrieval (capitulos, secoes, blocos consolidados)
   |    |--- Nivel 2: Micro retrieval (artigos, paragrafos, incisos, alineas)
   |
   +--> BM25 (in-memory, PT-BR stemming)
   +--> SQLite (conversas + control plane + document_nodes + document_summaries)
   +--> LLM (Groq / Anthropic)
```

## Estrutura do Projeto

```
app.py                â†’ FastAPI entry point, rotas, middleware ASGI
src/
  config.py           â†’ Settings via pydantic-settings (.env)
  llm.py              â†’ Chamadas LLM (Groq/Anthropic com fallback 429) e embeddings (fastembed/sentence-transformers)
  vectordb.py          â†’ FAISS persistente (data/chroma/)
  lexical.py          â†’ BM25 com stopwords e stemming PT-BR (RSLP)
  ingestion.py        → Parse + chunking juridico + embed + upsert + arvore juridica + macro indexing
  legal_tree.py       → Arvore juridica canonica (titulo > capitulo > secao > artigo) com fallback robusto para headings OCR/noisy
  summaries.py        → Resumos estruturais via LLM + fallback extrativo deterministico
  chat.py             → Retrieval hibrido + roteador de intencao + summary shortcut + fallback via artefato JSON da Prata + streaming SSE
  reranker.py         → Reranking deterministico + cross-encoder opcional
  guardrails.py       → Prompt injection, sanitizacao, rate limiting
  prompts.py          → Templates PT-BR com citacoes [N]
  domain_profiles.py  → Perfis geral/juridico/RH
  history.py          → Historico de conversas (SQLite)
  controlplane.py     → Workspaces, documentos, jobs, audit, document_nodes, document_summaries
  quality.py          → Utilidades de avaliacao
  evaluation.py       → Snapshot offline + avaliacao de sumarizacao estrutural
  observability.py    → Metricas e timers
  utils.py            â†’ Logging JSON, request_id, chunk ID deterministico
frontend/
  src/                â†’ React 19 + Vite + TypeScript
data/
  chroma/             → Indice vetorial FAISS persistente
  history.db          â†’ Historico SQLite
  docs/               â†’ Documentos brutos enviados
tests/                â†’ 223 testes unitarios
```

## RAG Juridica

Documentos juridicos (estatutos, contratos, regulamentos) recebem tratamento especial automatico:

**Chunking estrutural**: artigos sao a unidade semantica. Parent chunk = artigo inteiro; child chunks = paragrafos individuais quando o artigo e longo.

**Chunks contextualizados**: o embedding recebe metadados hierarquicos (`[Documento: X] [Caminho: Cap V > Art. 41]`), enquanto o texto cru e armazenado no indice FAISS.

**Preferencia por JSON estruturado (sem perder semantica)**: para PDFs processados no pipeline Prata, o embedding e enriquecido com contexto derivado do JSON de blocos (`pagina`, `section_hint`, `block_type`), mantendo o texto original para recall semantico.

**Retrieval em 2 niveis**:

### Nivel 1 — Macro retrieval

Capitulos, secoes e titulos indexados como chunks especiais no FAISS com `chunk_type=macro`. Permite:
- Buscar capitulos/secoes inteiros por similaridade semantica
- Recuperar blocos consolidados para perguntas de resumo
- Navegar pela arvore hierarquica do documento

### Nivel 2 — Micro retrieval

Artigos, paragrafos, incisos e alineas como chunks tradicionais. Permite:
- Perguntas especificas ("o que diz o art. 41?")
- Busca factual ("e proibido remunerar?")
- Localizacao de trechos exatos

**Retrieval em 2 estagios** (para ambos niveis):
1. Busca larga (BM25 + vetor + RRF com pesos dinamicos)
2. Reranking deterministico + cross-encoder opcional + expansao de contexto (parent + siblings + referencias)

**Calibracao de score no reranker**: o score final combina similaridade vetorial com cross-encoder normalizado (sigmoid), preservando faixa 0..1 e evitando cortes indevidos por `RETRIEVAL_MIN_SCORE`.

**Resgate estrutural (fallback de recall)**: quando a pergunta cita estrutura normativa (`capitulo`, `secao`, `artigo`), o pipeline suplementa o contexto com chunks cujo metadado estrutural casa com a referencia pedida (ex.: "capitulo 1").

**Atalho por artefato Prata para sumarizacao estrutural**: se `summary_structural` nao encontrar resumo persistido ou se os metadados vetoriais estiverem incompletos, o chat tenta resolver o escopo diretamente no artefato `data/processed/<doc>.json`, usando os blocos `section_header` para localizar `titulo`, `capitulo` e `secao`.

**Pesos dinamicos**: queries literais ("art. 37", "§ 2°") favorecem BM25 (0.65); queries conceituais ("e proibido remunerar?") favorecem embedding (0.65).

**Expansao de contexto**: ao recuperar um paragrafo (child), o sistema busca automaticamente o artigo pai (parent), paragrafos vizinhos (siblings) e artigos referenciados internamente.

Para ativar explicitamente: use `domain_profile=legal` no upload ou na query. O sistema tambem detecta documentos juridicos automaticamente (3+ indicadores de Art., CAPITULO, SECAO).

## Arvore Juridica Canonica

Na ingestao de documentos juridicos, o sistema gera automaticamente uma representacao hierarquica canonica:

```json
{
  "doc_id": "estatuto_social",
  "root": {
    "node_type": "documento",
    "children": [
      {
        "node_type": "titulo",
        "label": "TÍTULO I - DISPOSIÇÕES GERAIS",
        "children": [
          {
            "node_type": "capitulo",
            "label": "CAPÍTULO II - DOS OBJETIVOS SOCIAIS",
            "articles": ["Art. 3", "Art. 4", "Art. 5"],
            "children": [
              {
                "node_type": "secao",
                "label": "SEÇÃO I - DAS RESTRIÇÕES"
              }
            ]
          }
        ]
      }
    ]
  }
}
```

Cada no contem: texto consolidado, artigos cobertos, referencias internas, path hierarquico.

Os nos sao persistidos em SQLite (`document_nodes`) e indexados no FAISS como macro chunks.

## Resumos Pre-computados

Para cada capitulo e secao, o sistema gera automaticamente via LLM:

| Campo | Descricao |
|-------|-----------|
| `resumo_executivo` | 2-3 frases para leitura rapida por nao-juristas |
| `resumo_juridico` | Resumo tecnico com terminologia precisa |
| `pontos_chave` | Lista de pontos principais |
| `artigos_cobertos` | Artigos contidos no capitulo/secao |
| `obrigacoes` | Obrigacoes identificadas |
| `restricoes` | Vedacoes e restricoes |
| `definicoes` | Termos definidos no trecho |

Os resumos sao persistidos em SQLite (`document_summaries`) e usados como shortcut quando o roteador de intencao detecta `summary_structural`.

Cada resumo tambem carrega metadados de confiabilidade:
- `status=generated`: resumo estruturado valido retornado pelo LLM
- `status=fallback_only`: resumo gerado por fallback extrativo local
- `status=invalid`: payload descartado por schema/validacao
- `source_hash`, `source_text_length`, `generation_meta`: rastreabilidade do texto-base usado na geracao

## Roteador de Intencao

O sistema classifica cada pergunta antes de decidir o pipeline:

| Intencao | Exemplo | Pipeline |
|----------|---------|----------|
| `count_structural` | "Quantos capitulos tem no estatuto?" | Structure-first via artefato JSON da Prata |
| `list_structural` | "Quais artigos estao no Capitulo II?" | Structure-first com inventario por escopo |
| `locate_structural` | "Qual e o ultimo capitulo?" | Structure-first com localizacao por pagina/ordem |
| `contains_structural` | "O Capitulo V tem Secao III?" | Structure-first com validacao pai-filho |
| `summary_structural` | "Resuma o Capitulo II" | Resumo pre-computado + chunks de suporte |
| `question_structural` | "O que diz o Capitulo II?" | Retrieval focado no capitulo |
| `question_factual` | "E proibido remunerar?" | Retrieval hibrido padrao |
| `locate_excerpt` | "Art. 41" | Retrieval com boost estrutural |
| `comparison` | "Compare capitulo 1 e 2" | Retrieval multi-capitulo |

Quando uma intent `structure-first` e detectada (`count/list/locate/contains`), a resposta sai diretamente do artefato JSON da Prata, usando os blocos `section_header` e a ordem do documento como fonte de verdade.

Quando `summary_structural` e detectado, a ordem de preferencia e:
1. resumo persistido em `document_summaries`
2. resolucao direta do escopo no artefato JSON da Prata (`section_header`)
3. summary on-demand por escopo com fallback extrativo local

Isso elimina a dependencia exclusiva de retrieval fragmentado para perguntas como "Resuma o Capitulo II" e tambem cobre inventario/composicao estrutural com alta confiabilidade.

## Table-first para Planilhas

Quando a colecao possui dados tabulares estruturados, o chat tenta resolver perguntas analiticas antes da RAG textual.

Exemplos:
- "Qual e a renda do estado do Rio de Janeiro?"
- "Qual a media por estado?"
- "Top 10 clientes por renda"
- "Quais sao os estados UF da tabela?"
- "Quais sao as colunas da tabela?"
- "O que significa a coluna score_credito?"
- "Compare SP e RJ em renda media"

Para tabelas com perfil de catalogo/codigos, o sistema tambem cobre:
- "Qual e o procedimento com esse id 20017?"
- "Qual e o codigo da Visita Domiciliar?"
- "Qual a cobertura do 20017?"
- "Qual o prazo de autorizacao do 20017?"
- "Resuma o procedimento 20017"

Fluxo:
1. identificar que a pergunta e tabular/analitica
2. carregar o perfil semantico da base (`table_profile`, `column_profiles`, `value_catalog`)
3. gerar um plano estruturado (`intent`, `metric_column`, `aggregation`, `filters`, `group_by`, `expected_unit`)
4. validar semanticamente o plano antes da execucao
5. construir SQL seguro de forma deterministica
6. executar a consulta no store estruturado (`DuckDB` ou fallback `SQLite`)
7. renderizar a resposta com unidade correta (`R$`, `anos`, inteiro, lista, schema)
8. exibir no card a consulta analitica executada e um resumo do resultado

Quando a resposta vem de `table-first`, a interface nao trata o resultado como "5 fontes consultadas". Em vez disso, mostra:
- `consulta analitica executada`
- SQL auditavel da operacao
- preview resumido do resultado tabular

No frontend, a aba de ingestao tambem expõe a **inspecao semantica da base**:
- `table_profile` da colecao
- `column_profiles` com tipo fisico, tipo semantico, unidade, aliases e operacoes permitidas
- `value_catalog` com exemplos reais por coluna
- resumo do benchmark tabular
- sugestao automatica de modo antes do primeiro upload

### Arquitetura Semantica Tabular

A camada de planilhas/tabelas foi evoluida para um fluxo `schema-first` e `LLM-last`:

1. **Table profile layer**
   - persiste contexto da base, nome da tabela e sujeito de negocio
2. **Semantic catalog**
   - classifica cada coluna com:
     - `physical_type`
     - `semantic_type`
     - `role`
     - `unit`
     - `aliases`
     - `examples`
     - `allowed_operations`
3. **Planner estruturado**
   - produz intents como:
     - `tabular_aggregate`
     - `tabular_count`
     - `tabular_groupby`
     - `tabular_rank`
     - `tabular_distinct`
      - `tabular_schema`
      - `tabular_describe_column`
      - `catalog_lookup_by_id`
      - `catalog_lookup_by_title`
      - `catalog_field_lookup`
      - `catalog_record_summary`
      - `catalog_compare`
4. **Validator**
   - impede agregacoes incoerentes, como idade sendo renderizada como moeda
5. **SQL builder**
   - gera SQL seguro a partir do plano validado
6. **Semantic renderer**
   - transforma o resultado em resposta de negocio com unidade correta

### Catalog-first para Codigos e Procedimentos

Quando a tabela detectada tem perfil de catalogo (`table_type=catalog`), o sistema muda de comportamento:

1. trata colunas como `procedimento` e `codigo` como `identifier`, nunca como metrica
2. identifica uma coluna principal de titulo/descricao (`catalog_title`)
3. classifica atributos como cobertura, prazo e autorizacao como campos descritivos
4. responde por **lookup exato ou normalizado**, e nao por soma/media

Fluxo:
1. detectar que a tabela parece catalogo
2. escolher o modo `Catalogo/Codigos` no perfil semantico da base
3. rotear perguntas para intents como `catalog_lookup_by_id` e `catalog_field_lookup`
4. localizar o registro correto
5. responder em linguagem natural com os atributos relevantes

Na UI, o painel semantico da aba de ingestao agora mostra explicitamente:
- `Planilhas/Tabelas analiticas`
- `Catalogo/Codigos`

No chat, quando a resposta vem de `catalog-first`, a mensagem tambem recebe um badge visual:
- `Resposta por Catalogo/Codigos`

Isso evita respostas absurdas como media de `procedimento` e melhora PDFs tabulares de regras, codigos e rol assistencial.

### Persistencia de Metadados Tabulares

O control plane SQLite agora persiste:
- `table_profiles`
- `column_profiles`
- `value_catalog`
- `query_plans_log`

Isso permite auditoria, evolucao incremental do planner e benchmark tabular sem depender da RAG textual.

### Benchmark Tabular

Existe um benchmark tabular dedicado com dataset ouro em `data/eval/tabular_gold_dataset.json`, cobrindo casos como:
- agregacao por estado
- contagem com filtros
- distinct de UF
- schema da tabela
- descricao semantica de coluna
- comparacao executiva entre grupos
- lookup de procedimento por codigo
- lookup de codigo por descricao
- cobertura, prazo e autorizacao em tabelas de catalogo
- resumo de registro de catalogo

O endpoint `/evaluation/tabular` expõe esse snapshot para acompanhamento da qualidade do planner semântico.

### Refinamentos de UX Tabular

Para reduzir ambiguidade no uso por pessoas nao tecnicas, a experiencia tabular agora inclui:

1. **Sugestao pre-upload de modo da base**
   - heuristica local baseada em extensao e nome do arquivo
   - exemplos:
     - `.csv`, `.xlsx`, `.xls` -> `Planilhas/Tabelas`
     - nomes com `procedimento`, `rol`, `codigo`, `cobertura` -> subtipo `Catalogo/Codigos`
     - nomes com `estatuto`, `contrato`, `regulamento` -> `Juridico/Contratos`

2. **Sinalizacao visual no chat**
   - respostas de catalogo exibem badge explicito
   - ajuda o usuario a entender que a resposta veio de lookup estruturado, e nao de RAG textual

3. **Inspecao semantica com exemplos reais**
   - cada coluna pode exibir valores reais amostrados da base
   - isso facilita auditoria de semantic_type, aliases e operacoes permitidas

## Limpeza Estrutural

Alem da remocao de headers/footers e normalizacao de markdown, o pipeline aplica:

1. **Merge de linhas quebradas**: repara hifenizacao (`respons-\navel` → `responsavel`) e quebras mid-sentence
2. **Deduplicacao de paragrafos**: remove blocos duplicados (headers repetidos, secoes copiadas)
3. **Remocao de numeros de pagina orfaos**: linhas contendo apenas numeros
4. **Normalizacao de whitespace**: colapsa espacos e linhas em branco excessivos

Essa limpeza roda antes do chunking e melhora significativamente a qualidade dos embeddings.

## Embeddings PT-BR

| Dominio    | Modelo                                                          | Dimensao |
|------------|------------------------------------------------------------------|----------|
| Geral / RH | sentence-transformers/paraphrase-multilingual-mpnet-base-v2     | 768      |
| Juridico   | rufimelo/Legal-BERTimbau-sts-large                              | 1024     |
| Multilingual | intfloat/multilingual-e5-large                                | 1024     |

Configurar via `EMBEDDING_MODEL` no `.env`. Dimensao sempre consultada em runtime.

## Ingestion Pipeline (Bronze/Prata/Ouro)

1. **Bronze**: validar upload (collection/extensao/tamanho) e registrar recebimento
2. **Prata**: parsear e extrair conteudo (PDF/docling, openpyxl, csv etc.), gerar artefatos `.md` e `.json`
3. **Prata**: limpeza estrutural pesada (merge broken lines, dedup, normalize whitespace)
4. **Prata**: classificar documento e preparar texto limpo para chunking
5. **Ouro**: chunking estrutural/sentence-aware + embeddings
6. **Ouro**: delete de chunks antigos por `doc_id` + upsert no indice FAISS
7. **Ouro (juridico)**: construir arvore juridica canonica + indexar macro chunks (capitulos/secoes)
8. **Ouro (juridico)**: gerar resumos pre-computados por capitulo/secao via LLM
9. **Ouro (juridico)**: se o LLM falhar ou responder malformado, gerar summary extrativo deterministico (`fallback_only`)
10. **Ouro (juridico)**: persistir nos e resumos em SQLite (`document_nodes`, `document_summaries`)
11. Atualizar status/progresso da ingestao e inventario de documentos

Dois modos: `POST /ingest` (sincrono) e `POST /ingest/async` (background job).

Status usuais de pipeline: `queued`, `bronze_received`, `silver_processing`, `silver_extracted`, `gold_indexing`, `indexed`, `failed`.

No frontend, o polling da aba de ingestao e dinamico:
- rapido quando existem jobs ativos
- mais lento quando nao ha jobs em andamento

## Migracao / Reindexacao para FAISS

Depois da troca de backend vetorial, reindexe os documentos para preencher o novo indice:

```bash
python scripts/reindex_faiss.py --docs-dir data/docs --workspace-id default
```

Opcoes uteis:

```bash
# Reindexar apenas uma colecao
python scripts/reindex_faiss.py --collection geral

# Reindexar usando outro modelo de embedding
python scripts/reindex_faiss.py --embedding-model sentence-transformers/paraphrase-multilingual-mpnet-base-v2

# Limpar colecoes FAISS de destino antes do reindex
python scripts/reindex_faiss.py --clear-collections
```

### Quando reindexar obrigatoriamente

Reindexe a base quando houver qualquer mudanca que afete embedding/ranking, por exemplo:

- troca de `EMBEDDING_MODEL`
- upgrade/downgrade de versao do `fastembed` ou `sentence-transformers`
- mudanca de estrategia de pooling do modelo (ex.: aviso de CLS -> mean pooling)

Sem reindexacao, a base pode manter vetores antigos e degradar recall/precisao da RAG.

## Chat Pipeline

1. Resolver workspace e collection fisica
2. Validar e sanitizar question + history
3. Detectar prompt injection
4. **Classificar intencao da query** (`count_structural`, `list_structural`, `locate_structural`, `contains_structural`, `summary_structural`, `question_structural`, `question_factual`, `locate_excerpt`, `comparison`)
5. **Se `count/list/locate/contains_structural`**: resolver pela estrutura do artefato JSON da Prata e responder sem passar pela busca vetorial
6. **Se `summary_structural`**: buscar resumo persistido no SQLite
7. **Se nao encontrar**: tentar resolver `titulo/capitulo/secao` direto no artefato JSON da Prata
8. **Se encontrar escopo**: gerar summary on-demand (LLM ou fallback extrativo) → montar contexto com resumo + chunks de suporte → retornar
9. Classificar query (literal vs conceitual) para pesos dinamicos
10. Embed query
11. Busca vetorial + BM25 com fetch window 3x
12. Fusao RRF com pesos dinamicos
13. Reranking deterministico (lexical + metadata)
14. Cross-encoder reranking (opcional, `CROSS_ENCODER_ENABLED=true`)
15. Combinar score vetorial + score cross-encoder normalizado (0..1)
16. Expansao de contexto juridico (parent + siblings + referencias)
17. Resgate estrutural por metadados para "capitulo/secao/artigo"
18. Sanitizar contexto recuperado
19. Gerar resposta com LLM usando perfil de dominio
20. Garantir presenca de citacao [N]

Streaming via `POST /chat/message/stream` (SSE).

Os passos 5-7 sao a diferenca critica: "Resuma o Capitulo II" pode ser respondido por resumo persistido ou por resolucao direta do artefato JSON, sem depender apenas de retrieval fragmentado.

## LLM Fallback (Groq 429)

Quando o modelo primario do Groq retorna erro 429 (rate limit), o sistema tenta automaticamente:

1. **Modelos Groq alternativos**: `llama-3.1-8b-instant` â†’ `mixtral-8x7b-32768` â†’ `gemma2-9b-it`
2. **Anthropic** (se `ANTHROPIC_API_KEY` configurada): `claude-haiku-4-5-20251001`

O fallback funciona tanto para chamadas sincronas (`chat()`) quanto streaming (`chat_stream()`). Nenhuma configuracao extra necessaria â€” basta ter o `GROQ_API_KEY` configurado. Para habilitar o fallback para Anthropic, adicione `ANTHROPIC_API_KEY` no `.env`.

## Guardrails

- Validacao de question (max 2000 chars), collection (`[a-zA-Z0-9_-]` max 64), history (max 20 msgs)
- Deteccao de prompt injection: 13 padroes regex (PT-BR + EN)
- Sanitizacao de contexto: remove `<system>`, `[INST]`, "ignore instrucoes" dos chunks antes do LLM
- Rate limiting in-memory: chat 30 req/min, ingest 10 req/min por IP

## Main API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | health check |
| `GET` | `/settings` | configuracoes runtime |
| `GET` | `/observability` | contadores e timers |
| `GET` | `/evaluation/retrieval` | snapshot offline de qualidade |
| `GET` | `/workspaces` | listar workspaces |
| `POST` | `/workspaces` | criar workspace |
| `GET` | `/collections/available` | collections do workspace |
| `GET` | `/collections` | estatisticas das collections |
| `GET` | `/documents` | listar documentos indexados |
| `DELETE` | `/documents/{doc_id}` | deletar documento e chunks |
| `GET` | `/documents/{doc_id}/artifacts` | preview de artefatos Prata (`markdown_preview`, `json_preview`) |
| `GET` | `/documents/{doc_id}/artifacts/download` | download do artefato (`kind=markdown|json`) |
| `POST` | `/ingest` | ingestao sincrona |
| `POST` | `/ingest/async` | ingestao assincrona |
| `GET` | `/ingest/jobs` | listar jobs de ingestao |
| `GET` | `/ingest/jobs/{job_id}` | detalhes de job (inclui estagio e progresso) |
| `POST` | `/chat` | chat stateless |
| `POST` | `/chat/message` | chat persistente |
| `POST` | `/chat/message/stream` | chat com streaming SSE |
| `GET` | `/conversations` | listar/buscar conversas |
| `POST` | `/conversations` | criar conversa |
| `GET` | `/conversations/{id}/messages` | mensagens da conversa |
| `PATCH` | `/conversations/{id}` | renomear conversa |
| `DELETE` | `/conversations/{id}` | deletar conversa |
| `GET` | `/audit/events` | trail de auditoria |
| `DELETE` | `/admin/purge-conversations` | purge de conversas antigas |

## Tests

Testes cobrindo rotas, retrieval, reranking, ingestion, vectordb, llm (incluindo fallback), guardrails, history, prompts, evaluation, control plane, arvore juridica, resumos e avaliacao estrutural.

```bash
PYTHONPATH=. python -m pytest tests/ -x --tb=short -q
PYTHONPATH=. python -m pytest --cov=app --cov=src --cov-report=term-missing -q
```

## Avaliacao de Sumarizacao Estrutural

Metricas especificas para medir qualidade do resumo estrutural:

| Metrica | Descricao |
|---------|-----------|
| `structural_hit@1` | Achou o capitulo/secao correto? (1.0 ou 0.0) |
| `article_coverage` | Fracao dos artigos esperados cobertos pelo resumo |
| `summary_found_rate` | Taxa de perguntas em que um summary foi encontrado/gerado |
| `summary_valid_rate` | Taxa de summaries validos entre os encontrados |
| `summary_scope_accuracy` | O resumo veio do escopo estrutural correto? |
| `citation_scope_precision` | As citacoes/fontes pertencem ao mesmo escopo? |
| `fallback_success_rate` | Taxa de sucesso quando foi necessario fallback |

Dataset de avaliacao embutido com perguntas como:
- "Resuma o Capitulo II"
- "Quais as vedacoes do Capitulo X?"
- "Explique a Secao III em linguagem simples"
- "Liste os artigos cobertos pelo Capitulo 2"

## Limitacoes Conhecidas

- Rate limiting in-memory (single-process)
- BM25 in-memory, reconstruido quando necessario
- Busca de conversas usa LIKE, nao FTS
- Auth por API key, sem RBAC
- Ingestao async usa background tasks do FastAPI, nao worker queue dedicado
- Cross-encoder adiciona ~100-200ms de latencia por query
- Avaliacao de retrieval usa dataset de referencia embutido
- Resumos pre-computados dependem de chamada LLM na ingestao (custo adicional)
- Para resumir por estrutura sem reingestao, o sistema depende da existencia do artefato `data/processed/<doc>.json`
- Arvore juridica canonica ainda pode degradar em PDFs muito ruidosos, mas agora existe fallback por artefato JSON da Prata
- Roteador de intencao usa heuristicas regex (nao ML), pode errar em queries ambiguas

## Principios de Design

- Arquitetura flat, sem DI framework
- IDs de chunk deterministicos (SHA-256)
- FAISS sempre persistente
- Isolamento fisico de collections por workspace + modelo
- Prompting domain-aware com grounding conservador
- Reranking em 2 estagios antes da geracao
- Logging JSON estruturado com request_id
- Document-centric e structure-first para documentos juridicos
- Dois niveis de retrieval: macro (capitulos/secoes) e micro (artigos/paragrafos)
- Resumos pre-computados eliminam dependencia de retrieval fragmentado para sumarizacao
- Arvore juridica canonica como fonte de verdade para navegacao estrutural


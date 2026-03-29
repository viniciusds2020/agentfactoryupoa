"""PT-BR prompt templates for RAG chat."""
from __future__ import annotations

RAG_SYSTEM = """\
Voce e um assistente corporativo que conversa em portugues brasileiro de forma natural e direta.

Regras gerais:
- Use APENAS as informacoes dos trechos de contexto fornecidos.
- Se a informacao nao estiver no contexto, diga que nao encontrou nos documentos disponiveis.
- Se localizar uma secao fortemente relacionada, mas os trechos estiverem incompletos, diga explicitamente que o contexto parece parcial.
- Responda de forma conversacional, como um colega de trabalho experiente explicaria.
- Va direto ao ponto, sem repetir a pergunta nem listar fontes redundantes.
- Cite a fonte apenas uma vez ao final da informacao relevante, no formato [N].
- Nao liste multiplas fontes quando todas dizem a mesma coisa; cite apenas a mais completa.
- Nao use frases como "conforme informado nos documentos" ou "conforme mencionado em".
- Prefira frases curtas e naturais. Evite tom de relatorio.

Quando a pergunta pedir listas normativas, como direitos, deveres, obrigacoes, requisitos, vedacoes ou competencias:
- Liste os itens separadamente, sem misturar categorias diferentes.
- Preserve a organizacao por secao, artigo, paragrafo ou inciso quando essa estrutura estiver visivel no contexto.
- Se a pergunta comparar categorias irmas, como direitos versus deveres, responda cada grupo em blocos separados.

Resumos e visoes gerais:
Quando o usuario pedir para resumir, explicar ou descrever um capitulo, secao, parte ou o documento todo:
- Junte e sintetize as informacoes de TODOS os trechos fornecidos, mesmo que venham de paginas diferentes.
- Organize a resposta por temas ou topicos, nao por trecho.
- So faca um resumo quando os trechos recuperados representarem de fato o capitulo ou a secao pedida.
- Se o contexto trouxer apenas mencoes indiretas, historico de alteracoes, indice, referencias cruzadas ou trechos insuficientes, diga isso com clareza e NAO monte um pseudo-resumo.
- Se os trechos cobrem apenas parte do conteudo principal pedido, deixe explicito que o resumo e parcial.

Exemplo de resposta ruim:
"O valor e R$ 8.533,00, conforme informado nos documentos [1], [2], [3] e [6]."

Exemplo de resposta boa:
"O valor da nota fiscal e R$ 8.533,00 [1]."
"""

RAG_USER = """\
Contexto extraido dos documentos:
{context}

Pergunta do usuario:
{question}
"""

_DOMAIN_ADDENDUMS: dict[str, str] = {
    "general": (
        "Priorize respostas claras, objetivas e diretamente ancoradas no contexto. "
        "Quando houver ambiguidade, prefira a interpretacao mais conservadora."
    ),
    "legal": (
        "Trate o contexto como material juridico ou contratual. Destaque prazos, obrigacoes, "
        "multas, clausulas, vigencia e excecoes. Evite extrapolar alem do texto."
    ),
    "tabular": (
        "Trate o contexto como dados tabulares e estruturados. Priorize metricas, agregacoes, "
        "filtros, comparacoes e linguagem executiva. Quando a resposta vier de consulta analitica, "
        "seja direto, informe claramente o criterio adotado e evite narrar trechos como se fossem texto corrido."
    ),
    "hr": (
        "Trate o contexto como politica de pessoas e relacoes trabalhistas. Destaque beneficios, "
        "jornada, afastamentos, ferias, folha, elegibilidade e regras de RH."
    ),
}


def format_context(fused_results: list[dict]) -> str:
    """Format fused retrieval results as numbered context with source metadata.

    Groups parent and child chunks of the same article together so the LLM
    sees a coherent legal unit instead of scattered fragments.
    """
    # Group children under their parent's article key
    parent_keys_seen: dict[str, list[int]] = {}  # parent_key -> indices
    ordering: list[int | str] = []  # int = standalone index, str = parent_key (first occurrence)

    for i, item in enumerate(fused_results):
        meta = item.get("metadata", {})
        chunk_type = meta.get("chunk_type", "general")
        parent_key = meta.get("parent_key", "")

        if chunk_type in ("parent", "child") and parent_key:
            if parent_key not in parent_keys_seen:
                parent_keys_seen[parent_key] = []
                ordering.append(parent_key)
            parent_keys_seen[parent_key].append(i)
        else:
            ordering.append(i)

    parts: list[str] = []
    counter = 1

    for entry in ordering:
        if isinstance(entry, int):
            # Standalone chunk
            item = fused_results[entry]
            meta = item.get("metadata", {})
            label = _format_label(meta)
            parts.append(f"[{counter}] ({label})\nTrecho:\n{item['text']}")
            counter += 1
        else:
            # Grouped parent+children
            indices = parent_keys_seen[entry]
            # Sort: parent first, then children
            indices.sort(key=lambda idx: (0 if fused_results[idx].get("metadata", {}).get("chunk_type") == "parent" else 1))
            group_lines: list[str] = []
            first_meta = fused_results[indices[0]].get("metadata", {})
            label = _format_label(first_meta)
            caminho = first_meta.get("caminho_hierarquico", "")
            if caminho:
                label += f" | {caminho}"
            for idx in indices:
                group_lines.append(fused_results[idx]["text"])
            parts.append(f"[{counter}] ({label})\nTrecho:\n" + "\n\n".join(group_lines))
            counter += 1

    return "\n\n---\n\n".join(parts)


def _format_label(meta: dict) -> str:
    """Build a human-readable label from chunk metadata."""
    doc_id = meta.get("doc_id", "desconhecido")
    label = f"Documento: {doc_id}"
    page = meta.get("page_number")
    if page is not None:
        label += f", p. {page}"
    artigo = meta.get("artigo", "")
    if artigo:
        label += f", {artigo}"
    paragrafo = meta.get("paragrafo", "")
    if paragrafo:
        label += f", {paragrafo}"
    inciso = meta.get("inciso", "")
    if inciso:
        label += f", inciso {inciso}"
    return label


def build_rag_messages(context: str, question: str) -> list[dict]:
    return [
        {"role": "user", "content": RAG_USER.format(context=context, question=question)},
    ]


def get_rag_system(domain_profile: str | None = None) -> str:
    name = (domain_profile or "general").strip().lower()
    addendum = _DOMAIN_ADDENDUMS.get(name, _DOMAIN_ADDENDUMS["general"])
    return f"{RAG_SYSTEM}\n\nAjuste de dominio ({name}): {addendum}"


def list_domain_profiles() -> list[str]:
    return list(_DOMAIN_ADDENDUMS.keys())


# ── Query Expansion ──────────────────────────────────────────────────────────

QUERY_EXPANSION_PROMPT = """\
Reformule a pergunta abaixo de {n} formas diferentes, usando vocabulario tecnico \
e sinonimos do dominio corporativo brasileiro (trabalhista, juridico, RH).
Mantenha o significado original. Responda APENAS com as reformulacoes, uma por linha. \
Nao numere, nao adicione explicacoes.

Pergunta: {question}"""


def build_query_expansion_messages(question: str, n: int = 2) -> list[dict]:
    return [
        {"role": "user", "content": QUERY_EXPANSION_PROMPT.format(question=question, n=n)},
    ]


# ── HyDE (Hypothetical Document Embeddings) ─────────────────────────────────

HYDE_PROMPT = """\
Escreva um paragrafo curto (3-5 frases) que responderia a seguinte pergunta, \
usando linguagem formal corporativa brasileira. Nao invente dados especificos \
como numeros de artigos ou valores exatos. Apenas descreva o conceito geral.

Pergunta: {question}"""


def build_hyde_messages(question: str) -> list[dict]:
    return [
        {"role": "user", "content": HYDE_PROMPT.format(question=question)},
    ]

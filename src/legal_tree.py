"""Canonical legal document tree: structured JSON representation of legal documents.

Builds a hierarchical tree (Título > Capítulo > Seção > Artigo > Parágrafo > Inciso)
from parsed legal text. Each node contains consolidated text, metadata, and links
to children. This enables macro-retrieval (chapters/sections as units) and
pre-computed summaries.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from src.utils import get_logger

logger = get_logger(__name__)

# ── Regex for structural elements ──────────────────────────────────────────

_TITULO_RE = re.compile(
    r"^(T[IÍ]TULO\s+[IVXLCDM\d]+(?:\s*[-–—:\.]\s*.+)?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_CAPITULO_RE = re.compile(
    r"^(CAP[IÍ]TULO\s+[IVXLCDM\d]+(?:\s*[-–—:\.]\s*.+)?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_SECAO_RE = re.compile(
    r"^(SE[CÇ][AÃ]O\s+[IVXLCDM\d]+(?:\s*[-–—:\.]\s*.+)?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_ARTIGO_RE = re.compile(
    r"^(Art\.?\s*\d+[°ºª]?\s*[-–—.:]\s*)",
    re.IGNORECASE | re.MULTILINE,
)
_ARTIGO_LABEL_RE = re.compile(r"Art\.?\s*(\d+)", re.IGNORECASE)
_PARAGRAFO_RE = re.compile(
    r"^((?:§\s*\d+[°ºª]?\s*[-–—.:]\s*)|(?:Parágrafo\s+[Úú]nico\s*[-–—.:]\s*))",
    re.IGNORECASE | re.MULTILINE,
)
_INCISO_RE = re.compile(
    r"^((?:[IVXLCDM]+|\d+)\s*[-–—]\s+)",
    re.IGNORECASE | re.MULTILINE,
)
_ROMAN_RE = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)

_STRUCTURAL_NUM_RE = re.compile(
    r"(?:T[IÍ]TULO|CAP[IÍ]TULO|SE[CÇ][AÃ]O)\s+([IVXLCDM]+|\d+)",
    re.IGNORECASE,
)


def _normalize_structural_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized.replace("–", "-").replace("—", "-")


def _extract_numeral(header: str) -> str:
    """Extract the numeral (Roman or Arabic) from a structural header.

    Matches the numeral that follows the structural keyword (TÍTULO, CAPÍTULO, SEÇÃO),
    avoiding false matches on letters within the keyword itself.
    """
    m = _STRUCTURAL_NUM_RE.search(header)
    if not m:
        m = re.search(
            r"(?:TITULO|CAPITULO|SECAO)\s+([IVXLCDM]+|\d+)",
            _normalize_structural_text(header),
            re.IGNORECASE,
        )
    return m.group(1).upper() if m else ""


def _node_id(doc_id: str, node_type: str, label: str) -> str:
    """Deterministic node ID."""
    raw = f"{doc_id}::{node_type}::{label}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class LegalNode:
    """A node in the legal document tree."""
    id: str
    node_type: str  # "documento" | "titulo" | "capitulo" | "secao" | "artigo" | "paragrafo" | "inciso"
    label: str  # e.g., "CAPÍTULO II - DOS OBJETIVOS SOCIAIS"
    numeral: str  # e.g., "II", "2"
    text: str  # consolidated clean text of this node (including children)
    page_start: int | None = None
    page_end: int | None = None
    parent_id: str = ""
    path: str = ""  # hierarchical path: "Título I > Capítulo II > Seção III"
    children: list[LegalNode] = field(default_factory=list)
    articles: list[str] = field(default_factory=list)  # article numbers contained
    internal_refs: list[str] = field(default_factory=list)  # cross-references

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "id": self.id,
            "node_type": self.node_type,
            "label": self.label,
            "numeral": self.numeral,
            "text": self.text[:500] + ("..." if len(self.text) > 500 else ""),
            "text_length": len(self.text),
            "page_start": self.page_start,
            "page_end": self.page_end,
            "parent_id": self.parent_id,
            "path": self.path,
            "articles": self.articles,
            "internal_refs": self.internal_refs,
            "children": [c.to_dict() for c in self.children],
        }

    def full_text(self) -> str:
        """Return the full consolidated text of this node and all children."""
        return self.text


@dataclass
class LegalTree:
    """Full canonical representation of a legal document."""
    doc_id: str
    doc_name: str
    root: LegalNode
    node_index: dict[str, LegalNode] = field(default_factory=dict)

    def get_node(self, node_id: str) -> LegalNode | None:
        return self.node_index.get(node_id)

    def get_chapters(self) -> list[LegalNode]:
        """Return all chapter-level nodes."""
        return [n for n in self.node_index.values() if n.node_type == "capitulo"]

    def get_sections(self) -> list[LegalNode]:
        """Return all section-level nodes."""
        return [n for n in self.node_index.values() if n.node_type == "secao"]

    def get_titles(self) -> list[LegalNode]:
        """Return all title-level nodes."""
        return [n for n in self.node_index.values() if n.node_type == "titulo"]

    def get_macro_nodes(self) -> list[LegalNode]:
        """Return all macro-level nodes (título, capítulo, seção) for indexing."""
        macro_types = {"titulo", "capitulo", "secao"}
        return [n for n in self.node_index.values() if n.node_type in macro_types]

    def find_by_label(self, query: str) -> list[LegalNode]:
        """Find nodes whose label matches query (case-insensitive partial match)."""
        q = query.strip().lower()
        return [n for n in self.node_index.values() if q in n.label.lower()]

    def find_by_numeral(self, node_type: str, numeral: str) -> LegalNode | None:
        """Find node by type and numeral (e.g., capitulo + "II")."""
        numeral_upper = numeral.strip().upper()
        for n in self.node_index.values():
            if n.node_type == node_type and n.numeral.upper() == numeral_upper:
                return n
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "root": self.root.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ── Tree builder ───────────────────────────────────────────────────────────

def _find_all_matches(pattern: re.Pattern, text: str) -> list[tuple[int, str]]:
    """Return (offset, header_text) for all matches of pattern."""
    return [(m.start(), m.group(1).strip()) for m in pattern.finditer(text)]


def _find_structural_markers_fallback(text: str) -> list[tuple[int, str, str]]:
    """Fallback line-based marker detection for OCR/encoding-noisy headings."""
    markers: list[tuple[int, str, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        normalized = _normalize_structural_text(stripped).upper()
        if re.match(r"^TITULO\s+([IVXLCDM]+|\d+)\b", normalized):
            markers.append((offset, "titulo", stripped))
        elif re.match(r"^CAPITULO\s+([IVXLCDM]+|\d+)\b", normalized):
            markers.append((offset, "capitulo", stripped))
        elif re.match(r"^SECAO\s+([IVXLCDM]+|\d+)\b", normalized):
            markers.append((offset, "secao", stripped))
        offset += len(line)
    return markers


def _text_between(text: str, start: int, end: int) -> str:
    """Extract and clean text between two offsets."""
    return text[start:end].strip()


def _count_articles(text: str) -> list[str]:
    """Extract article numbers from a text block."""
    return [m.group(1) for m in _ARTIGO_LABEL_RE.finditer(text)]


def _extract_refs(text: str) -> list[str]:
    """Extract internal article references."""
    refs = set()
    for m in re.finditer(r"art(?:igo)?\.?\s*(\d+)", text, re.IGNORECASE):
        refs.add(f"Art. {m.group(1)}")
    return sorted(refs)


def build_legal_tree(text: str, doc_id: str, doc_name: str = "") -> LegalTree:
    """Build a canonical legal tree from cleaned document text.

    The tree has up to 6 levels: documento > título > capítulo > seção > artigo > parágrafo/inciso.
    Each node stores consolidated text for its entire scope.
    """
    doc_name = doc_name or doc_id

    # Collect all structural markers with offsets
    titulos = _find_all_matches(_TITULO_RE, text)
    capitulos = _find_all_matches(_CAPITULO_RE, text)
    secoes = _find_all_matches(_SECAO_RE, text)

    # Build all structural markers sorted by offset
    markers: list[tuple[int, str, str]] = []  # (offset, type, header)
    for offset, header in titulos:
        markers.append((offset, "titulo", header))
    for offset, header in capitulos:
        markers.append((offset, "capitulo", header))
    for offset, header in secoes:
        markers.append((offset, "secao", header))
    if not markers:
        markers.extend(_find_structural_markers_fallback(text))
    markers.sort(key=lambda x: x[0])
    deduped: list[tuple[int, str, str]] = []
    seen_marker_keys: set[tuple[int, str]] = set()
    for marker in markers:
        key = (marker[0], marker[1])
        if key in seen_marker_keys:
            continue
        seen_marker_keys.add(key)
        deduped.append(marker)
    markers = deduped

    node_index: dict[str, LegalNode] = {}

    # Root node
    root_id = _node_id(doc_id, "documento", doc_id)
    root = LegalNode(
        id=root_id,
        node_type="documento",
        label=doc_name,
        numeral="",
        text=text,
        path=doc_name,
        articles=_count_articles(text),
        internal_refs=_extract_refs(text),
    )
    node_index[root_id] = root

    if not markers:
        # No structural markers — flat document, return root only
        tree = LegalTree(doc_id=doc_id, doc_name=doc_name, root=root, node_index=node_index)
        return tree

    # Build nodes for each structural section
    # Strategy: iterate markers, each marker owns text until the next marker of same or higher level
    level_order = {"titulo": 1, "capitulo": 2, "secao": 3}

    # Create nodes with text spans
    nodes_list: list[tuple[LegalNode, int]] = []  # (node, level)

    for i, (offset, ntype, header) in enumerate(markers):
        # Find end of this section: next marker of same or higher level, or end of text
        level = level_order[ntype]
        end = len(text)
        for j in range(i + 1, len(markers)):
            next_level = level_order[markers[j][1]]
            if next_level <= level:
                end = markers[j][0]
                break

        section_text = _text_between(text, offset, end)
        numeral = _extract_numeral(header)
        nid = _node_id(doc_id, ntype, header)

        node = LegalNode(
            id=nid,
            node_type=ntype,
            label=header,
            numeral=numeral,
            text=section_text,
            articles=_count_articles(section_text),
            internal_refs=_extract_refs(section_text),
        )
        node_index[nid] = node
        nodes_list.append((node, level))

    # Build parent-child relationships
    # Use a stack-based approach: maintain current parent at each level
    parent_stack: list[LegalNode] = [root]
    level_stack: list[int] = [0]

    for node, level in nodes_list:
        # Pop stack until we find a parent at a higher level
        while level_stack and level_stack[-1] >= level:
            parent_stack.pop()
            level_stack.pop()

        parent = parent_stack[-1] if parent_stack else root
        node.parent_id = parent.id

        # Build path
        if parent.node_type == "documento":
            node.path = node.label
        else:
            node.path = f"{parent.path} > {node.label}"

        parent.children.append(node)

        # Push this node as potential parent
        parent_stack.append(node)
        level_stack.append(level)

    tree = LegalTree(doc_id=doc_id, doc_name=doc_name, root=root, node_index=node_index)

    logger.info(
        f"Legal tree built: {len(titulos)} títulos, {len(capitulos)} capítulos, "
        f"{len(secoes)} seções, {len(tree.get_macro_nodes())} macro nodes"
    )
    return tree

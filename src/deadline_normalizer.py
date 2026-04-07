from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass


@dataclass
class DeadlineInfo:
    raw_text: str
    days: int | None
    unit: str
    urgency: str
    requires_authorization: bool
    faixa: str

    def to_dict(self) -> dict:
        return asdict(self)


_NUMBER_WORDS = {
    "um": 1,
    "uma": 1,
    "dois": 2,
    "duas": 2,
    "tres": 3,
    "quatro": 4,
    "cinco": 5,
    "seis": 6,
    "sete": 7,
    "oito": 8,
    "nove": 9,
    "dez": 10,
}


def _norm(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def _extract_days(text: str) -> tuple[int | None, str]:
    norm = _norm(text)
    if "imediato" in norm:
        return 0, "imediato"
    m = re.search(r"ate\s+(\d{1,3})\s*(?:\([^)]+\))?\s*(dias uteis|dias|horas)", norm)
    if m:
        return int(m.group(1)), m.group(2).replace(" ", "_")
    m = re.search(r"ate\s+\(?([a-z]+)\)?\s*(dias uteis|dias|horas)", norm)
    if m and m.group(1) in _NUMBER_WORDS:
        return _NUMBER_WORDS[m.group(1)], m.group(2).replace(" ", "_")
    return None, "texto"


def _urgency_and_band(days: int | None, unit: str, norm: str) -> tuple[str, str]:
    if "sem cobertura" in norm:
        return "sem_cobertura", "Sem Cobertura"
    if "nao autoriza" in norm:
        return "nao_autoriza", "Nao autoriza"
    if unit == "imediato" or "imediato" in norm:
        return "imediato", "Imediato"
    if days is None:
        return "desconhecido", "Desconhecido"
    if days <= 3:
        return "curto", "Ate 3 dias"
    if days <= 5:
        return "curto", "Ate 5 dias"
    if days <= 10:
        return "medio", "Ate 10 dias"
    return "longo", "Acima de 10 dias"


def normalize_deadline(text: str) -> DeadlineInfo:
    raw = str(text or "").strip()
    norm = _norm(raw)
    days, unit = _extract_days(raw)
    urgency, faixa = _urgency_and_band(days, unit, norm)
    requires_authorization = False
    if "necessita autoriz" in norm or "autorizacao obrigatoria" in norm:
        requires_authorization = True
    if urgency in {"nao_autoriza", "sem_cobertura"}:
        requires_authorization = False
    return DeadlineInfo(
        raw_text=raw,
        days=days,
        unit=unit,
        urgency=urgency,
        requires_authorization=requires_authorization,
        faixa=faixa,
    )

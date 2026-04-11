from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.medical_catalog.schemas import CatalogIngestionConfig, ProcedureRecord, SourceReference
from src.medical_catalog.text import compact_whitespace, normalize_key


def _canonical_aliases(config: CatalogIngestionConfig) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for canonical in config.expected_columns:
        mapping[normalize_key(canonical)] = canonical
    for canonical, aliases in config.aliases.items():
        mapping[normalize_key(canonical)] = canonical
        for alias in aliases:
            mapping[normalize_key(alias)] = canonical
    return mapping


def normalize_dataframe(df: pd.DataFrame, config: CatalogIngestionConfig) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=config.expected_columns)

    work = df.copy()
    work = work.dropna(how="all").dropna(axis=1, how="all")
    work = work.fillna("")
    work = work.astype(str)
    work = work.apply(lambda column: column.map(compact_whitespace))
    work = work[(work != "").any(axis=1)]
    if work.empty:
        return pd.DataFrame(columns=config.expected_columns)

    if list(work.columns) == list(range(len(work.columns))):
        first_row = [compact_whitespace(v) for v in work.iloc[0].tolist()]
        aliases = _canonical_aliases(config)
        recognized = sum(1 for value in first_row if normalize_key(value) in aliases)
        if recognized >= max(2, len(first_row) // 2):
            work.columns = first_row
            work = work.iloc[1:].reset_index(drop=True)

    aliases = _canonical_aliases(config)
    renamed: dict[str, str] = {}
    for column in work.columns:
        renamed[column] = aliases.get(normalize_key(str(column)), normalize_key(str(column)))
    work = work.rename(columns=renamed)

    for column in config.expected_columns:
        if column not in work.columns:
            work[column] = ""

    ordered = work[config.expected_columns].copy()
    ordered = ordered[(ordered != "").any(axis=1)].reset_index(drop=True)
    return ordered


def dataframe_to_records(
    df: pd.DataFrame,
    config: CatalogIngestionConfig,
    source_file: str,
    page_number: int,
    extractor: str,
) -> list[ProcedureRecord]:
    records: list[ProcedureRecord] = []
    if df.empty:
        return records

    for _, row in df.iterrows():
        fields = {column: compact_whitespace(str(row.get(column, ""))) for column in config.expected_columns}
        codigo = fields.get(config.code_column, "")
        descricao = fields.get(config.description_column, "")
        if not codigo or not descricao:
            continue
        searchable_text = " ".join(
            part
            for part in [
                f"{col} {fields.get(col, '')}".strip()
                for col in config.search_columns
                if fields.get(col, "")
            ]
        )
        records.append(
            ProcedureRecord(
                codigo=codigo,
                descricao=descricao,
                fields=fields,
                source=SourceReference(
                    source_file=source_file,
                    page_number=page_number,
                    extractor=extractor,
                    excerpt=searchable_text[:240],
                ),
                searchable_text=searchable_text,
            )
        )
    return records


def deduplicate_records(records: Iterable[ProcedureRecord]) -> list[ProcedureRecord]:
    deduped: dict[str, ProcedureRecord] = {}
    for record in records:
        key = record.codigo
        if key not in deduped:
            deduped[key] = record
            continue
        if len(record.searchable_text) > len(deduped[key].searchable_text):
            deduped[key] = record
    return list(deduped.values())

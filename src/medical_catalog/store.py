from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from src.medical_catalog.schemas import CatalogIndex


def storage_root() -> Path:
    configured = os.getenv("MEDICAL_CATALOG_STORAGE_DIR")
    return Path(configured) if configured else Path("data/medical_catalog")


class CatalogStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or storage_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def catalog_dir(self, catalog_id: str) -> Path:
        path = self.root / catalog_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def index_path(self, catalog_id: str) -> Path:
        return self.catalog_dir(catalog_id) / "index.json"

    def save(self, index: CatalogIndex) -> Path:
        payload = index.model_dump(mode="json")
        path = self.index_path(index.catalog_id)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load(self, catalog_id: str) -> CatalogIndex:
        path = self.index_path(catalog_id)
        if not path.exists():
            raise FileNotFoundError(f"Catálogo '{catalog_id}' ainda não foi indexado.")
        return CatalogIndex.model_validate_json(path.read_text(encoding="utf-8"))

    def exists(self, catalog_id: str) -> bool:
        return self.index_path(catalog_id).exists()

    @staticmethod
    def now_iso() -> str:
        return datetime.now(UTC).isoformat()

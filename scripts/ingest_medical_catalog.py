from __future__ import annotations

import argparse
import json

from src.medical_catalog.schemas import ReindexRequest
from src.medical_catalog.service import get_medical_catalog_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestão de catálogo médico em PDF.")
    parser.add_argument("--catalog-id", default="default")
    parser.add_argument("--pdf-path", required=True)
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args()

    service = get_medical_catalog_service()
    result = service.reindex(
        ReindexRequest(
            catalog_id=args.catalog_id,
            pdf_path=args.pdf_path,
            config_path=args.config_path,
        )
    )
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from table_designer.blueprints import BlueprintExporter
from table_designer.geometry import TableGeometryBuilder
from table_designer.params import TableParams, validate_params


def main(output_dir: Path) -> None:
    params = TableParams()
    messages = validate_params(params)
    if messages:
        for msg in messages:
            print(msg)
        raise SystemExit(1)

    parts = TableGeometryBuilder(params).build_parts()
    BlueprintExporter(params).export(parts, output_dir)
    print(f"Wrote blueprints to {output_dir}")


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "blueprints"
    main(out_dir)

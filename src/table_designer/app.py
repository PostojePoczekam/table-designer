from __future__ import annotations

from pathlib import Path

from .blueprints import BlueprintExporter
from .geometry import TableGeometryBuilder
from .obj_exporter import ObjExporter
from .params import TableParams, validate_params


def main() -> None:
    params = TableParams()

    msgs = validate_params(params)
    for msg in msgs:
        print(msg)
    if any(msg.startswith("ERROR") for msg in msgs):
        raise SystemExit("Aborted: invalid parameters.")

    out_dir = Path("out_table")
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = TableGeometryBuilder(params).build_parts()

    obj_path = out_dir / "table_yup.obj"
    ObjExporter(params).export(parts, obj_path)
    print(f"OBJ saved: {obj_path.resolve()}")

    BlueprintExporter(params).export(parts, out_dir)
    print(f"Blueprints saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

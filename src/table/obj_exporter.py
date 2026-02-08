from __future__ import annotations

from pathlib import Path

import trimesh

from .params import TableParams
from .transforms import rotation_matrix_x, translation_matrix


class ObjExporter:
    def __init__(self, params: TableParams) -> None:
        self._params = params

    def export(self, parts: dict[str, trimesh.Trimesh], out_path: Path) -> Path:
        merged = trimesh.util.concatenate(list(parts.values()))
        merged = merged.copy()
        merged.process(validate=True)

        merged_win = self._transform_for_windows_viewer(merged)
        merged_win.export(out_path)
        return out_path

    def _transform_for_windows_viewer(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Convert our Z-up model to Y-up, and put floor at Y=0.
        Z-up -> Y-up : rotate -90 deg about X
        Then translate +table_height in Y so floor is at 0.
        """
        m = mesh.copy()
        r = rotation_matrix_x(-90.0)
        t = translation_matrix(0.0, self._params.table_height, 0.0)
        m.apply_transform(r)
        m.apply_transform(t)
        return m

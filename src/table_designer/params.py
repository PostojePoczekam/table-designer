from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TableParams:
    # Units: mm
    top_length: float = 2800.0
    top_width: float = 1200.0
    top_thickness: float = 35.0

    table_height: float = 750.0  # "floor" at z = -table_height, top surface at z = 0 (Z-up)

    # Legs (square, tapered)
    leg_top_size: float = 80.0
    leg_bottom_size: float = 60.0
    leg_offset_x: float = 140.0  # distance from tabletop edge to leg axis (X)
    leg_offset_y: float = 120.0  # distance from tabletop edge to leg axis (Y)
    leg_splay_x_deg: float = -3.0
    leg_splay_y_deg: float = -2.0

    # Apron (oskrzynia)
    apron_height: float = 120.0
    apron_thickness: float = 22.0
    apron_top_gap: float = 0.0
    apron_inset: float = 110.0
    apron_miter: bool = False  # not implemented yet; kept for API
    leg_to_apron_clearance: float = 4.0

    def derived(self) -> dict[str, float]:
        leg_length = self.table_height - self.top_thickness
        return {"leg_length": leg_length}


def validate_params(p: TableParams) -> list[str]:
    msgs: list[str] = []
    d = p.derived()
    if d["leg_length"] <= 0:
        msgs.append("ERROR: table_height must be greater than top_thickness.")

    inner_l = p.top_length - 2 * (p.apron_inset + p.apron_thickness)
    inner_w = p.top_width - 2 * (p.apron_inset + p.apron_thickness)
    if inner_l <= 0 or inner_w <= 0:
        msgs.append("ERROR: apron inset/thickness leave no inner clearance.")

    return msgs

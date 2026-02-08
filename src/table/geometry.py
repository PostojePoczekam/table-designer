from __future__ import annotations

import numpy as np
import trimesh

from .params import TableParams
from .transforms import rotation_matrix_x, rotation_matrix_y, translation_matrix


class TableGeometryBuilder:
    def __init__(self, params: TableParams) -> None:
        self._params = params

    def build_parts(self) -> dict[str, trimesh.Trimesh]:
        p = self._params
        leg_length = p.derived()["leg_length"]

        parts: dict[str, trimesh.Trimesh] = {}

        # Z-up: top surface at z=0, underside at z=-top_thickness, floor at z=-table_height
        parts["tabletop"] = self._make_box(
            center_xyz=(0.0, 0.0, -p.top_thickness / 2.0),
            size_xyz=(p.top_length, p.top_width, p.top_thickness),
        )

        apron_top_z = -p.top_thickness - p.apron_top_gap
        apron_center_z = apron_top_z - p.apron_height / 2.0

        outer_l = p.top_length - 2 * p.apron_inset
        outer_w = p.top_width - 2 * p.apron_inset

        long_size = (outer_l, p.apron_thickness, p.apron_height)
        y_long = (outer_w / 2.0) - (p.apron_thickness / 2.0)
        parts["apron_front"] = self._make_box((0.0, +y_long, apron_center_z), long_size)
        parts["apron_back"] = self._make_box((0.0, -y_long, apron_center_z), long_size)

        short_size = (p.apron_thickness, outer_w, p.apron_height)
        x_short = (outer_l / 2.0) - (p.apron_thickness / 2.0)
        parts["apron_right"] = self._make_box((+x_short, 0.0, apron_center_z), short_size)
        parts["apron_left"] = self._make_box((-x_short, 0.0, apron_center_z), short_size)

        leg_proto = self._make_tapered_square_leg(leg_length, p.leg_top_size, p.leg_bottom_size)

        leg_top_z = -p.top_thickness
        x0 = (p.top_length / 2.0) - p.leg_offset_x
        y0 = (p.top_width / 2.0) - p.leg_offset_y

        corners = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]
        for i, (sx, sy) in enumerate(corners, start=1):
            ax = p.leg_splay_x_deg * sx
            ay = p.leg_splay_y_deg * sy

            leg = leg_proto.copy()
            r = rotation_matrix_y(ax) @ rotation_matrix_x(-ay)
            t = translation_matrix(sx * x0, sy * y0, leg_top_z)

            leg.apply_transform(r)
            leg.apply_transform(t)
            parts[f"leg_{i}"] = leg

        return parts

    @staticmethod
    def _make_box(
        center_xyz: tuple[float, float, float],
        size_xyz: tuple[float, float, float],
    ) -> trimesh.Trimesh:
        mesh = trimesh.creation.box(extents=size_xyz)
        mesh.apply_translation(center_xyz)
        return mesh

    @staticmethod
    def _make_tapered_square_leg(
        leg_length: float,
        top_size: float,
        bottom_size: float,
    ) -> trimesh.Trimesh:
        """
        Leg in local coordinates (Z-up):
          - top square centered at (0,0,0)
          - bottom square centered at (0,0,-leg_length)
        """
        ht = top_size / 2.0
        hb = bottom_size / 2.0
        z0 = 0.0
        z1 = -leg_length

        v = np.array(
            [
                [-ht, -ht, z0],
                [ht, -ht, z0],
                [ht, ht, z0],
                [-ht, ht, z0],
                [-hb, -hb, z1],
                [hb, -hb, z1],
                [hb, hb, z1],
                [-hb, hb, z1],
            ],
            dtype=float,
        )

        faces = []
        faces += [[0, 1, 2], [0, 2, 3]]
        faces += [[4, 6, 5], [4, 7, 6]]
        side_quads = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        for a, b, c, d in side_quads:
            faces += [[a, b, c], [a, c, d]]

        return trimesh.Trimesh(vertices=v, faces=np.array(faces, dtype=int), process=True)

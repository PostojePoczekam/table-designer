from __future__ import annotations

from pathlib import Path
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from .params import TableParams


class BlueprintExporter:
    def __init__(self, params: TableParams) -> None:
        self._params = params

    def export(self, parts: dict[str, trimesh.Trimesh], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._save_projection(parts, out_dir / "top.png", out_dir / "top.svg", view="top")
        self._save_projection(parts, out_dir / "front.png", out_dir / "front.svg", view="front")
        self._save_projection(parts, out_dir / "side.png", out_dir / "side.svg", view="side")
        self._save_detail_corner(parts, out_dir / "detail_corner.png", out_dir / "detail_corner.svg")

    def export_png_buffers(self, parts: dict[str, trimesh.Trimesh]) -> dict[str, bytes]:
        return {
            "top": self._render_png(parts, view="top"),
            "front": self._render_png(parts, view="front"),
            "side": self._render_png(parts, view="side"),
            "detail_corner": self._render_png(parts, view="detail_corner"),
        }

    def _save_projection(
        self,
        parts: dict[str, trimesh.Trimesh],
        out_png: Path,
        out_svg: Path,
        view: str,
    ) -> None:
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)

        silhouettes = self._silhouettes_by_part(parts, view=view)
        for geometry in silhouettes:
            self._plot_silhouette(ax, geometry)

        if view == "top":
            self._add_top_dimensions(ax, parts)
        elif view == "front":
            self._add_front_dimensions(ax, parts)
        elif view == "side":
            self._add_side_dimensions(ax, parts)
            self._add_leg_angle_indicator(ax, parts, view="front")
        elif view == "side":
            self._add_leg_angle_indicator(ax, parts, view="side")

        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        fig.tight_layout(pad=0)

        fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0)
        fig.savefig(out_svg, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def _save_detail_corner(
        self,
        parts: dict[str, trimesh.Trimesh],
        out_png: Path,
        out_svg: Path,
    ) -> None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)

        silhouettes = self._silhouettes_by_part(parts, view="top")
        for geometry in silhouettes:
            self._plot_silhouette(ax, geometry)
        self._add_leg_face_fills(ax, parts, view="top")
        self._add_leg_face_outlines(ax, parts, view="top")
        self._add_leg_face_dimensions(ax, parts, view="top")
        self._add_detail_apron_dimensions(ax, parts)

        x_min, x_max, y_min, y_max = self._bounds_for_geometries(silhouettes)
        detail_span = min(600.0, (x_max - x_min) * 0.5, (y_max - y_min) * 0.8)
        margin = 40.0
        ax.set_xlim(x_max - detail_span - margin, x_max + margin)
        ax.set_ylim(y_max - detail_span - margin, y_max + margin)

        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        fig.tight_layout(pad=0)

        fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0)
        fig.savefig(out_svg, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def _render_png(self, parts: dict[str, trimesh.Trimesh], view: str) -> bytes:
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)

        if view == "detail_corner":
            silhouettes = self._silhouettes_by_part(parts, view="top")
        else:
            silhouettes = self._silhouettes_by_part(parts, view=view)

        for geometry in silhouettes:
            self._plot_silhouette(ax, geometry)

        if view == "top":
            self._add_top_dimensions(ax, parts)
        elif view == "front":
            self._add_front_dimensions(ax, parts)
            self._add_leg_angle_indicator(ax, parts, view="front")
        elif view == "side":
            self._add_side_dimensions(ax, parts)
            self._add_leg_angle_indicator(ax, parts, view="side")
        elif view == "detail_corner":
            self._add_leg_face_fills(ax, parts, view="top")
            self._add_leg_face_outlines(ax, parts, view="top")
            self._add_leg_face_dimensions(ax, parts, view="top")
            self._add_detail_apron_dimensions(ax, parts)
            x_min, x_max, y_min, y_max = self._bounds_for_geometries(silhouettes)
            detail_span = min(600.0, (x_max - x_min) * 0.5, (y_max - y_min) * 0.8)
            margin = 40.0
            ax.set_xlim(x_max - detail_span - margin, x_max + margin)
            ax.set_ylim(y_max - detail_span - margin, y_max + margin)

        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        fig.tight_layout(pad=0)

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return buf.getvalue()

    def _add_top_dimensions(self, ax: plt.Axes, parts: dict[str, trimesh.Trimesh]) -> None:
        x_min, x_max, y_min, y_max = self._bounds_for_parts({"tabletop": parts["tabletop"]}, view="top")
        offset = 80.0
        self._add_dimension(
            ax,
            (x_min, y_min - offset),
            (x_max, y_min - offset),
            f"{self._params.top_length:.0f} mm",
            text_offset=(0.0, -30.0),
        )
        self._add_dimension(
            ax,
            (x_min - offset, y_min),
            (x_min - offset, y_max),
            f"{self._params.top_width:.0f} mm",
            text_offset=(-30.0, 0.0),
        )

    def _add_front_dimensions(self, ax: plt.Axes, parts: dict[str, trimesh.Trimesh]) -> None:
        x_min, x_max, z_min, z_max = self._bounds_for_parts(parts, view="front")
        offset = 80.0
        apron_offset = 160.0
        self._add_dimension(
            ax,
            (x_min - offset, z_min),
            (x_min - offset, z_max),
            f"{self._params.table_height:.0f} mm",
            text_offset=(-30.0, 0.0),
        )
        apron_top = -self._params.top_thickness - self._params.apron_top_gap
        apron_bottom = apron_top - self._params.apron_height
        self._add_dimension(
            ax,
            (x_min - apron_offset, apron_bottom),
            (x_min - apron_offset, apron_top),
            f"{self._params.apron_height:.0f} mm",
            text_offset=(-30.0, 0.0),
        )

    def _add_side_dimensions(self, ax: plt.Axes, parts: dict[str, trimesh.Trimesh]) -> None:
        x_min, x_max, z_min, z_max = self._bounds_for_parts(parts, view="side")
        offset = 80.0
        apron_offset = 160.0
        self._add_dimension(
            ax,
            (x_min - offset, z_min),
            (x_min - offset, z_max),
            f"{self._params.table_height:.0f} mm",
            text_offset=(-30.0, 0.0),
        )
        apron_top = -self._params.top_thickness - self._params.apron_top_gap
        apron_bottom = apron_top - self._params.apron_height
        self._add_dimension(
            ax,
            (x_min - apron_offset, apron_bottom),
            (x_min - apron_offset, apron_top),
            f"{self._params.apron_height:.0f} mm",
            text_offset=(-30.0, 0.0),
        )

    def _add_dimension(
        self,
        ax: plt.Axes,
        start: tuple[float, float],
        end: tuple[float, float],
        text: str,
        text_offset: tuple[float, float],
        zorder: float = 5.0,
    ) -> None:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="<->", color="black", linewidth=0.8, mutation_scale=8),
            zorder=zorder,
            clip_on=False,
        )
        text_x = (start[0] + end[0]) / 2.0 + text_offset[0]
        text_y = (start[1] + end[1]) / 2.0 + text_offset[1]
        ax.text(text_x, text_y, text, fontsize=8, ha="center", va="center", zorder=zorder, clip_on=False)

    def _bounds_for_geometry(
        self,
        geometry: Polygon | MultiPolygon,
    ) -> tuple[float, float, float, float]:
        x_min, y_min, x_max, y_max = geometry.bounds
        return float(x_min), float(x_max), float(y_min), float(y_max)

    def _bounds_for_geometries(
        self,
        geometries: list[Polygon | MultiPolygon],
    ) -> tuple[float, float, float, float]:
        if not geometries:
            return 0.0, 0.0, 0.0, 0.0
        bounds = [self._bounds_for_geometry(geom) for geom in geometries]
        x_min = min(b[0] for b in bounds)
        x_max = max(b[1] for b in bounds)
        y_min = min(b[2] for b in bounds)
        y_max = max(b[3] for b in bounds)
        return x_min, x_max, y_min, y_max

    def _bounds_for_parts(
        self,
        parts: dict[str, trimesh.Trimesh],
        view: str,
    ) -> tuple[float, float, float, float]:
        projected = []
        for mesh in parts.values():
            projected.append(self._project_points(mesh.vertices, view=view))
        stacked = np.vstack(projected)
        x_min = float(stacked[:, 0].min())
        x_max = float(stacked[:, 0].max())
        y_min = float(stacked[:, 1].min())
        y_max = float(stacked[:, 1].max())
        return x_min, x_max, y_min, y_max

    def _silhouettes_by_part(
        self,
        parts: dict[str, trimesh.Trimesh],
        view: str,
    ) -> list[Polygon | MultiPolygon]:
        silhouettes: list[Polygon | MultiPolygon] = []
        for mesh in parts.values():
            vertices = mesh.vertices
            faces = mesh.faces
            projected = self._project_points(vertices, view=view)

            triangles: list[Polygon] = []
            for face in faces:
                tri = projected[face]
                triangles.append(Polygon(tri))

            silhouettes.append(unary_union(triangles))

        return silhouettes

    def _add_leg_face_outlines(
        self,
        ax: plt.Axes,
        parts: dict[str, trimesh.Trimesh],
        view: str,
    ) -> None:
        for name, mesh in parts.items():
            if not name.startswith("leg_"):
                continue
            top_poly, bottom_poly = self._leg_face_polygons(mesh, view=view)
            for poly in (top_poly, bottom_poly):
                if poly is None:
                    continue
                x, y = poly.exterior.xy
                ax.plot(x, y, color="black", linewidth=0.8, zorder=4.0)

    def _add_leg_face_fills(
        self,
        ax: plt.Axes,
        parts: dict[str, trimesh.Trimesh],
        view: str,
    ) -> None:
        for name, mesh in parts.items():
            if not name.startswith("leg_"):
                continue
            top_poly, bottom_poly = self._leg_face_polygons(mesh, view=view)
            for poly in (top_poly, bottom_poly):
                if poly is None:
                    continue
                x, y = poly.exterior.xy
                ax.fill(x, y, color="white", linewidth=0, zorder=3.0)

    def _add_leg_face_dimensions(
        self,
        ax: plt.Axes,
        parts: dict[str, trimesh.Trimesh],
        view: str,
    ) -> None:
        leg_entries: list[tuple[float, Polygon | None, Polygon | None]] = []
        for name, mesh in parts.items():
            if not name.startswith("leg_"):
                continue
            top_poly, bottom_poly = self._leg_face_polygons(mesh, view=view)
            if top_poly is None:
                continue
            score = float(top_poly.centroid.x + top_poly.centroid.y)
            leg_entries.append((score, top_poly, bottom_poly))

        if not leg_entries:
            return

        _, top_poly, bottom_poly = max(leg_entries, key=lambda item: item[0])
        top_offset = 20.0
        bottom_offset = 60.0

        self._add_face_dimensions(
            ax,
            top_poly,
            self._params.leg_top_size,
            offset=top_offset,
            position="left_bottom",
        )
        if bottom_poly is not None:
            self._add_face_dimensions(
                ax,
                bottom_poly,
                self._params.leg_bottom_size,
                offset=bottom_offset,
                position="right_top",
            )

    def _add_face_dimensions(
        self,
        ax: plt.Axes,
        poly: Polygon | None,
        size_mm: float,
        offset: float,
        position: str,
    ) -> None:
        if poly is None:
            return
        x_min, x_max, y_min, y_max = self._bounds_for_geometry(poly)
        label = f"{size_mm:.0f} mm"
        if position == "left_bottom":
            self._add_dimension(
                ax,
                (x_min, y_min - offset),
                (x_max, y_min - offset),
                label,
                text_offset=(0.0, -12.0),
            )
            self._add_dimension(
                ax,
                (x_min - offset, y_min),
                (x_min - offset, y_max),
                label,
                text_offset=(-12.0, 0.0),
            )
        elif position == "right_top":
            self._add_dimension(
                ax,
                (x_min, y_max + offset),
                (x_max, y_max + offset),
                label,
                text_offset=(0.0, 12.0),
            )
            self._add_dimension(
                ax,
                (x_max + offset, y_min),
                (x_max + offset, y_max),
                label,
                text_offset=(12.0, 0.0),
            )
        else:
            raise ValueError("position must be 'left_bottom' or 'right_top'")

    def _add_detail_apron_dimensions(self, ax: plt.Axes, parts: dict[str, trimesh.Trimesh]) -> None:
        tabletop = parts.get("tabletop")
        if tabletop is None:
            return

        x_min, x_max, y_min, y_max = self._bounds_for_parts({"tabletop": tabletop}, view="top")
        outer_l = self._params.top_length - 2 * self._params.apron_inset
        outer_w = self._params.top_width - 2 * self._params.apron_inset
        if outer_l <= 0 or outer_w <= 0:
            return

        x_apron_outer = outer_l / 2.0
        y_apron_outer = outer_w / 2.0

        leg_top_poly = self._detail_leg_top_face(parts, view="top")
        if leg_top_poly is None:
            return
        leg_x_min, leg_x_max, leg_y_min, leg_y_max = self._bounds_for_geometry(leg_top_poly)

        detail_span = min(600.0, (x_max - x_min) * 0.5, (y_max - y_min) * 0.8)
        margin = 40.0
        x_left = x_max - detail_span - margin
        y_bottom = y_max - detail_span - margin

        mid_y = (y_bottom + leg_y_min) / 2.0
        mid_x = (x_left + leg_x_min) / 2.0

        # Inset X (tabletop edge to apron outer face).
        self._add_dimension(
            ax,
            (x_apron_outer, mid_y),
            (x_max, mid_y),
            f"{self._params.apron_inset:.0f} mm",
            text_offset=(0.0, 12.0),
        )
        # Inset Y (tabletop edge to apron outer face).
        self._add_dimension(
            ax,
            (mid_x, y_apron_outer),
            (mid_x, y_max),
            f"{self._params.apron_inset:.0f} mm",
            text_offset=(12.0, 0.0),
        )

    def _detail_leg_top_face(self, parts: dict[str, trimesh.Trimesh], view: str) -> Polygon | None:
        leg_entries: list[tuple[float, Polygon | None]] = []
        for name, mesh in parts.items():
            if not name.startswith("leg_"):
                continue
            top_poly, _ = self._leg_face_polygons(mesh, view=view)
            if top_poly is None:
                continue
            score = float(top_poly.centroid.x + top_poly.centroid.y)
            leg_entries.append((score, top_poly))

        if not leg_entries:
            return None
        _, top_poly = max(leg_entries, key=lambda item: item[0])
        return top_poly

    def _add_leg_angle_indicator(self, ax: plt.Axes, parts: dict[str, trimesh.Trimesh], view: str) -> None:
        leg = self._angle_leg_mesh(parts, view=view)
        if leg is None:
            return

        top_center, bottom_center = self._leg_axis_centers(leg)
        if top_center is None or bottom_center is None:
            return

        top_2d = self._project_points(top_center[None, :], view=view)[0]
        bottom_2d = self._project_points(bottom_center[None, :], view=view)[0]

        ref_top = np.array([bottom_2d[0], top_2d[1]])
        ref_bottom = np.array([bottom_2d[0], bottom_2d[1]])

        ax.plot(
            [ref_bottom[0], ref_top[0]],
            [ref_bottom[1], ref_top[1]],
            color="black",
            linewidth=1.0,
            linestyle=(0, (3, 3)),
            zorder=6.0,
            clip_on=False,
        )
        ax.plot(
            [bottom_2d[0], top_2d[0]],
            [bottom_2d[1], top_2d[1]],
            color="black",
            linewidth=1.0,
            linestyle=(0, (3, 3)),
            zorder=6.0,
            clip_on=False,
        )

        angle = self._params.leg_splay_x_deg if view == "front" else self._params.leg_splay_y_deg
        label = rf"${abs(angle):.0f}^\circ$"
        y_min = min(bottom_2d[1], top_2d[1])
        text_offset = np.array([12.0, -18.0])
        ax.text(
            bottom_2d[0] + text_offset[0],
            y_min + text_offset[1],
            label,
            fontsize=12,
            ha="left",
            va="bottom",
            zorder=7.0,
            clip_on=False,
        )

    def _detail_leg_mesh(self, parts: dict[str, trimesh.Trimesh], view: str) -> trimesh.Trimesh | None:
        leg_entries: list[tuple[float, trimesh.Trimesh]] = []
        for name, mesh in parts.items():
            if not name.startswith("leg_"):
                continue
            top_poly, _ = self._leg_face_polygons(mesh, view="top")
            if top_poly is None:
                continue
            score = float(top_poly.centroid.x + top_poly.centroid.y)
            leg_entries.append((score, mesh))

        if not leg_entries:
            return None
        _, mesh = max(leg_entries, key=lambda item: item[0])
        return mesh

    def _angle_leg_mesh(self, parts: dict[str, trimesh.Trimesh], view: str) -> trimesh.Trimesh | None:
        leg_entries: list[tuple[float, float, trimesh.Trimesh]] = []
        for name, mesh in parts.items():
            if not name.startswith("leg_"):
                continue
            centroid = mesh.vertices.mean(axis=0)
            proj = self._project_points(centroid[None, :], view=view)[0]
            leg_entries.append((abs(float(proj[0])), float(proj[0]), mesh))

        if not leg_entries:
            return None
        _, _, mesh = min(leg_entries, key=lambda item: (item[0], item[1]))
        return mesh

    def _leg_axis_centers(self, mesh: trimesh.Trimesh) -> tuple[np.ndarray | None, np.ndarray | None]:
        vertices = mesh.vertices
        if vertices.shape[0] < 8:
            return None, None

        centered = vertices - vertices.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        projections = centered @ axis

        order = np.argsort(projections)
        top_pts = vertices[order[-4:]]
        bottom_pts = vertices[order[:4]]
        return top_pts.mean(axis=0), bottom_pts.mean(axis=0)
    def _leg_face_polygons(
        self,
        mesh: trimesh.Trimesh,
        view: str,
    ) -> tuple[Polygon | None, Polygon | None]:
        vertices = mesh.vertices
        if vertices.shape[0] < 8:
            return None, None

        centered = vertices - vertices.mean(axis=0)
        # Use PCA to estimate the leg axis, then pick the extreme face vertices.
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        projections = centered @ axis

        order = np.argsort(projections)
        top_pts = vertices[order[-4:]]
        bottom_pts = vertices[order[:4]]

        top_projected = self._project_points(top_pts, view=view)
        bottom_projected = self._project_points(bottom_pts, view=view)

        top_poly = Polygon(top_projected).convex_hull
        bottom_poly = Polygon(bottom_projected).convex_hull
        return top_poly, bottom_poly

    @staticmethod
    def _plot_silhouette(
        ax: plt.Axes,
        geometry: Polygon | MultiPolygon,
        zorder: float = 1.0,
    ) -> None:
        if isinstance(geometry, Polygon):
            geometries = [geometry]
        else:
            geometries = list(geometry.geoms)

        for poly in geometries:
            x, y = poly.exterior.xy
            ax.plot(x, y, color="black", linewidth=1.0, zorder=zorder)

    @staticmethod
    def _project_points(points: np.ndarray, view: str) -> np.ndarray:
        if view == "top":
            return points[:, [0, 1]]
        if view == "front":
            return points[:, [0, 2]]
        if view == "side":
            return points[:, [1, 2]]
        raise ValueError("view must be one of: top, front, side")

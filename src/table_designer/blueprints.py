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
        self._add_leg_face_outlines(ax, parts, view="top")
        self._add_leg_face_dimensions(ax, parts, view="top")

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
        elif view == "detail_corner":
            self._add_leg_face_outlines(ax, parts, view="top")
            self._add_leg_face_dimensions(ax, parts, view="top")
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
        self._add_dimension(
            ax,
            (x_min - offset, z_min),
            (x_min - offset, z_max),
            f"{self._params.table_height:.0f} mm",
            text_offset=(-30.0, 0.0),
        )

    def _add_dimension(
        self,
        ax: plt.Axes,
        start: tuple[float, float],
        end: tuple[float, float],
        text: str,
        text_offset: tuple[float, float],
    ) -> None:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="<->", color="black", linewidth=0.8),
        )
        text_x = (start[0] + end[0]) / 2.0 + text_offset[0]
        text_y = (start[1] + end[1]) / 2.0 + text_offset[1]
        ax.text(text_x, text_y, text, fontsize=8, ha="center", va="center")

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
                ax.plot(x, y, color="black", linewidth=0.8)

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
        )
        if bottom_poly is not None:
            self._add_face_dimensions(
                ax,
                bottom_poly,
                self._params.leg_bottom_size,
                offset=bottom_offset,
            )

    def _add_face_dimensions(
        self,
        ax: plt.Axes,
        poly: Polygon | None,
        size_mm: float,
        offset: float,
    ) -> None:
        if poly is None:
            return
        x_min, x_max, y_min, y_max = self._bounds_for_geometry(poly)
        label = f"{size_mm:.0f} mm"
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
    def _plot_silhouette(ax: plt.Axes, geometry: Polygon | MultiPolygon) -> None:
        if isinstance(geometry, Polygon):
            geometries = [geometry]
        else:
            geometries = list(geometry.geoms)

        for poly in geometries:
            x, y = poly.exterior.xy
            ax.plot(x, y, color="black", linewidth=1.0)

    @staticmethod
    def _project_points(points: np.ndarray, view: str) -> np.ndarray:
        if view == "top":
            return points[:, [0, 1]]
        if view == "front":
            return points[:, [0, 2]]
        if view == "side":
            return points[:, [1, 2]]
        raise ValueError("view must be one of: top, front, side")

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import trimesh

if __package__ is None:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    SRC_ROOT = PROJECT_ROOT / "src"
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

from table.blueprints import BlueprintExporter
from table.geometry import TableGeometryBuilder
from table.params import TableParams, validate_params
from table.pdf_exporter import PdfExporter


STRINGS = {
    "en": {
        "title": "Table Builder",
        "generate": "Generate",
        "info": "Adjust parameters and click Generate.",
        "view3d": "3D View",
        "blueprints": "Blueprints",
        "detail_corner": "Detail corner",
        "export_pdf": "Export PDF",
        "pdf_name": "table_blueprint.pdf",
        "language": "Language",
    },
    "pl": {
        "title": "Generator stolu",
        "generate": "Generuj",
        "info": "Zmien parametry i kliknij Generuj.",
        "view3d": "Widok 3D",
        "blueprints": "Rzuty",
        "detail_corner": "Detal naroznika",
        "export_pdf": "Eksportuj PDF",
        "pdf_name": "projekt_stolu.pdf",
        "language": "Jezyk",
    },
}


def _t(language: str, key: str) -> str:
    return STRINGS.get(language, STRINGS["en"]).get(key, key)


def _mesh_to_plotly(mesh: trimesh.Trimesh) -> go.Figure:
    vertices = mesh.vertices
    faces = mesh.faces
    edges = mesh.edges_unique
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#c9a67a",
                opacity=1.0,
                flatshading=True,
            ),
            _edge_trace(vertices, edges),
        ]
    )
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


def _edge_trace(vertices: trimesh.caching.TrackedArray, edges: trimesh.caching.TrackedArray) -> go.Scatter3d:
    x: list[float] = []
    y: list[float] = []
    z: list[float] = []

    for a, b in edges:
        x.extend([vertices[a, 0], vertices[b, 0], None])
        y.extend([vertices[a, 1], vertices[b, 1], None])
        z.extend([vertices[a, 2], vertices[b, 2], None])

    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line=dict(color="#3f2f20", width=1),
        hoverinfo="skip",
        name="edges",
    )


def _plotly_to_png(fig: go.Figure) -> bytes | None:
    try:
        return pio.to_image(fig, format="png", scale=2)
    except Exception:
        return None


def main() -> None:
    st.set_page_config(page_title="Table Builder", layout="wide")
    language = st.selectbox("Language", options=["English", "Polski"], index=0, format_func=str)
    language_code = "pl" if language == "Polski" else "en"
    st.title(_t(language_code, "title"))

    with st.form("params"):
        col1, col2, col3 = st.columns(3)
        with col1:
            top_length = st.number_input("Top length (mm)", value=2800.0, min_value=100.0, step=10.0)
            top_width = st.number_input("Top width (mm)", value=1200.0, min_value=100.0, step=10.0)
            top_thickness = st.number_input("Top thickness (mm)", value=35.0, min_value=1.0, step=1.0)
            table_height = st.number_input("Table height (mm)", value=750.0, min_value=100.0, step=10.0)
        with col2:
            leg_top_size = st.number_input("Leg top size (mm)", value=80.0, min_value=10.0, step=1.0)
            leg_bottom_size = st.number_input("Leg bottom size (mm)", value=60.0, min_value=10.0, step=1.0)
            leg_offset_x = st.number_input("Leg offset X (mm)", value=140.0, min_value=0.0, step=5.0)
            leg_offset_y = st.number_input("Leg offset Y (mm)", value=120.0, min_value=0.0, step=5.0)
            leg_splay_x_deg = st.number_input("Leg splay X (deg)", value=-3.0, step=0.5)
        with col3:
            leg_splay_y_deg = st.number_input("Leg splay Y (deg)", value=-2.0, step=0.5)
            apron_height = st.number_input("Apron height (mm)", value=120.0, min_value=1.0, step=1.0)
            apron_thickness = st.number_input("Apron thickness (mm)", value=22.0, min_value=1.0, step=1.0)
            apron_inset = st.number_input("Apron inset (mm)", value=110.0, min_value=0.0, step=5.0)
            apron_top_gap = st.number_input("Apron top gap (mm)", value=0.0, min_value=0.0, step=1.0)

        submitted = st.form_submit_button(_t(language_code, "generate"))

    if not submitted:
        st.info(_t(language_code, "info"))
        return

    params = TableParams(
        top_length=top_length,
        top_width=top_width,
        top_thickness=top_thickness,
        table_height=table_height,
        leg_top_size=leg_top_size,
        leg_bottom_size=leg_bottom_size,
        leg_offset_x=leg_offset_x,
        leg_offset_y=leg_offset_y,
        leg_splay_x_deg=leg_splay_x_deg,
        leg_splay_y_deg=leg_splay_y_deg,
        apron_height=apron_height,
        apron_thickness=apron_thickness,
        apron_top_gap=apron_top_gap,
        apron_inset=apron_inset,
    )

    messages = validate_params(params)
    if messages:
        for msg in messages:
            st.error(msg)
        return

    parts = TableGeometryBuilder(params).build_parts()
    merged = trimesh.util.concatenate(list(parts.values()))
    merged = merged.copy()
    merged.process(validate=True)

    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.subheader(_t(language_code, "view3d"))
        fig = _mesh_to_plotly(merged)
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        st.subheader(_t(language_code, "blueprints"))
        buffers = BlueprintExporter(params).export_png_buffers(parts)
        st.image(buffers["top"], caption="Top")
        st.image(buffers["front"], caption="Front")
        st.image(buffers["side"], caption="Side")
        st.image(buffers["detail_corner"], caption=_t(language_code, "detail_corner"))

    pdf_data = PdfExporter(params, language_code).export(
        blueprints=buffers,
        three_d_image=_plotly_to_png(fig),
    )
    st.download_button(
        _t(language_code, "export_pdf"),
        data=pdf_data,
        file_name=_t(language_code, "pdf_name"),
        mime="application/pdf",
    )


if __name__ == "__main__":
    main()

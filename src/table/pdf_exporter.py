from __future__ import annotations

from io import BytesIO
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)

from .params import TableParams


class PdfExporter:
    def __init__(self, params: TableParams, language: str) -> None:
        self._params = params
        self._language = language
        self._styles = getSampleStyleSheet()

    def export(
        self,
        blueprints: dict[str, bytes],
        three_d_image: bytes | None,
    ) -> bytes:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm)
        story = []

        story.append(Paragraph(self._text("title"), self._styles["Title"]))
        story.append(Spacer(1, 6 * mm))
        story.extend(self._params_table())
        story.append(Spacer(1, 6 * mm))

        story.append(Paragraph(self._text("blueprints"), self._styles["Heading2"]))
        story.append(Spacer(1, 4 * mm))
        story.extend(self._blueprint_images(blueprints))

        if three_d_image is not None:
            story.append(Spacer(1, 6 * mm))
            story.append(Paragraph(self._text("view3d"), self._styles["Heading2"]))
            story.append(Spacer(1, 4 * mm))
            story.append(self._image_from_bytes(three_d_image, width=170 * mm))

        doc.build(story)
        return buffer.getvalue()

    def _params_table(self) -> list:
        data = [
            [self._text("param"), self._text("value")],
            [self._text("top_length"), f"{self._params.top_length:.0f} mm"],
            [self._text("top_width"), f"{self._params.top_width:.0f} mm"],
            [self._text("top_thickness"), f"{self._params.top_thickness:.0f} mm"],
            [self._text("table_height"), f"{self._params.table_height:.0f} mm"],
            [self._text("leg_top_size"), f"{self._params.leg_top_size:.0f} mm"],
            [self._text("leg_bottom_size"), f"{self._params.leg_bottom_size:.0f} mm"],
            [self._text("leg_offset_x"), f"{self._params.leg_offset_x:.0f} mm"],
            [self._text("leg_offset_y"), f"{self._params.leg_offset_y:.0f} mm"],
            [self._text("leg_splay_x_deg"), f"{self._params.leg_splay_x_deg:.1f} deg"],
            [self._text("leg_splay_y_deg"), f"{self._params.leg_splay_y_deg:.1f} deg"],
            [self._text("apron_height"), f"{self._params.apron_height:.0f} mm"],
            [self._text("apron_thickness"), f"{self._params.apron_thickness:.0f} mm"],
            [self._text("apron_inset"), f"{self._params.apron_inset:.0f} mm"],
            [self._text("apron_top_gap"), f"{self._params.apron_top_gap:.0f} mm"],
        ]
        table = Table(data, hAlign="LEFT", colWidths=[60 * mm, 60 * mm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (1, 1), (1, -1), "LEFT"),
                ]
            )
        )
        return [table]

    def _blueprint_images(self, blueprints: dict[str, bytes]) -> Iterable:
        order = [
            ("top", self._text("top_view")),
            ("front", self._text("front_view")),
            ("side", self._text("side_view")),
            ("detail_corner", self._text("detail_corner")),
        ]
        items = []
        for key, label in order:
            data = blueprints.get(key)
            if data is None:
                continue
            items.append(Paragraph(label, self._styles["Heading3"]))
            items.append(Spacer(1, 2 * mm))
            items.append(self._image_from_bytes(data, width=170 * mm))
            items.append(Spacer(1, 4 * mm))
        return items

    @staticmethod
    def _image_from_bytes(data: bytes, width: float) -> Image:
        img = Image(BytesIO(data))
        img.drawWidth = width
        img.drawHeight = img.imageHeight * (width / img.imageWidth)
        return img

    def _text(self, key: str) -> str:
        strings = {
            "en": {
                "title": "Table Blueprint",
                "blueprints": "Blueprints",
                "view3d": "3D View",
                "param": "Parameter",
                "value": "Value",
                "top_length": "Top length",
                "top_width": "Top width",
                "top_thickness": "Top thickness",
                "table_height": "Table height",
                "leg_top_size": "Leg top size",
                "leg_bottom_size": "Leg bottom size",
                "leg_offset_x": "Leg offset X",
                "leg_offset_y": "Leg offset Y",
                "leg_splay_x_deg": "Leg splay X",
                "leg_splay_y_deg": "Leg splay Y",
                "apron_height": "Apron height",
                "apron_thickness": "Apron thickness",
                "apron_inset": "Apron inset",
                "apron_top_gap": "Apron top gap",
                "top_view": "Top view",
                "front_view": "Front view",
                "side_view": "Side view",
                "detail_corner": "Detail corner",
            },
            "pl": {
                "title": "Projekt stołu",
                "blueprints": "Rzuty",
                "view3d": "Widok 3D",
                "param": "Parametr",
                "value": "Wartosc",
                "top_length": "Dlugosc blatu",
                "top_width": "Szerokosc blatu",
                "top_thickness": "Grubosc blatu",
                "table_height": "Wysokosc stolu",
                "leg_top_size": "Noga - wymiar gora",
                "leg_bottom_size": "Noga - wymiar dol",
                "leg_offset_x": "Odsuniecie nogi X",
                "leg_offset_y": "Odsuniecie nogi Y",
                "leg_splay_x_deg": "Rozstaw nog X",
                "leg_splay_y_deg": "Rozstaw nog Y",
                "apron_height": "Wysokosc oskrzyni",
                "apron_thickness": "Grubosc oskrzyni",
                "apron_inset": "Wciecie oskrzyni",
                "apron_top_gap": "Szczelina u gory",
                "top_view": "Rzut z gory",
                "front_view": "Rzut z przodu",
                "side_view": "Rzut z boku",
                "detail_corner": "Detal naroznika",
            },
        }
        return strings.get(self._language, strings["en"]).get(key, key)

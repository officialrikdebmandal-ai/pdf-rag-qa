#!/usr/bin/env python3
"""Generate the bundled CC BY 4.0 sample PDF (and the Figure 1 PNG).

The figure is embedded as a raster image so that the exact survival
percentages exist only in pixels, not in the PDF text layer. That is
intentional: it gives the retrieval tool a documented failure case.

Build-time extras (not required to run pdfqa):

    pip install reportlab pillow
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import tempfile

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
PDF_PATH = SAMPLES / "brain_mri_overview.pdf"

# These numbers appear ONLY in the raster figure, not in the PDF body text.
FIGURE_VALUES = {
    "Glioblastoma": 7,
    "WHO grade 1\nmeningioma": 88,
    "Pituitary\nadenoma": 93,
}


def _font(size: int):
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_figure(path: Path) -> Path:
    width, height = 1200, 700
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = _font(28)
    label_font = _font(20)
    axis_font = _font(18)

    margin_left, margin_right = 120, 60
    margin_top, margin_bottom = 90, 140
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    draw.text(
        (width // 2, 30),
        "Figure 1. Illustrative five-year survival by tumor type",
        fill="black",
        font=title_font,
        anchor="mt",
    )

    # Axes
    origin = (margin_left, margin_top + plot_h)
    draw.line([origin, (margin_left, margin_top)], fill="black", width=3)
    draw.line(
        [origin, (margin_left + plot_w, margin_top + plot_h)],
        fill="black",
        width=3,
    )
    for pct in range(0, 101, 20):
        y = margin_top + plot_h - int(plot_h * (pct / 100))
        draw.line([(margin_left - 8, y), (margin_left, y)], fill="black", width=2)
        draw.text((margin_left - 16, y), f"{pct}%", fill="black", font=axis_font, anchor="rm")

    bars = list(FIGURE_VALUES.items())
    slot = plot_w / len(bars)
    bar_w = int(slot * 0.5)
    palette = [(188, 80, 80), (80, 120, 180), (80, 150, 110)]
    for i, (label, value) in enumerate(bars):
        cx = margin_left + int((i + 0.5) * slot)
        bar_h = int(plot_h * (value / 100))
        x0 = cx - bar_w // 2
        y0 = margin_top + plot_h - bar_h
        x1 = cx + bar_w // 2
        y1 = margin_top + plot_h
        draw.rectangle([x0, y0, x1, y1], fill=palette[i], outline="black", width=2)
        # Value label sits in the image pixels only.
        draw.text((cx, y0 - 12), f"{value}%", fill="black", font=label_font, anchor="mb")
        label_y = y1 + 16
        for line_i, line in enumerate(label.split("\n")):
            draw.text(
                (cx, label_y + line_i * 24),
                line,
                fill="black",
                font=label_font,
                anchor="mt",
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "PNG")
    return path


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "CoverTitle",
            parent=styles["Title"],
            fontName="Times-Bold",
            fontSize=18,
            leading=22,
            spaceAfter=8,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            "CoverMeta",
            parent=styles["Normal"],
            fontName="Times-Italic",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#333333"),
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            "BodyJust",
            parent=styles["Normal"],
            fontName="Times-Roman",
            fontSize=11,
            leading=15,
            alignment=TA_JUSTIFY,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            "Heading1Custom",
            parent=styles["Heading1"],
            fontName="Times-Bold",
            fontSize=14,
            leading=18,
            spaceBefore=8,
            spaceAfter=8,
            textColor=colors.HexColor("#1a1a1a"),
        )
    )
    styles.add(
        ParagraphStyle(
            "Caption",
            parent=styles["Normal"],
            fontName="Times-Italic",
            fontSize=9,
            leading=12,
            alignment=TA_LEFT,
            spaceBefore=4,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            "TableCell",
            parent=styles["Normal"],
            fontName="Times-Roman",
            fontSize=8,
            leading=10,
        )
    )
    styles.add(
        ParagraphStyle(
            "TableHeader",
            parent=styles["Normal"],
            fontName="Times-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
        )
    )
    return styles


def _p(text: str, style) -> Paragraph:
    return Paragraph(text, style)


def build_pdf(pdf_path: Path, figure_path: Path) -> Path:
    styles = _styles()
    cell = styles["TableCell"]
    header = styles["TableHeader"]

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="A Brief Overview of Brain Tumors and MRI Imaging",
        author="Rikdeb Mandal",
        subject="Sample CC BY 4.0 document for retrieval-based PDF question answering",
    )

    story = []
    story.append(_p("A Brief Overview of Brain Tumors and MRI Imaging", styles["CoverTitle"]))
    story.append(
        _p(
            "Sample document for retrieval-based question answering. "
            "Author: Rikdeb Mandal. Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).",
            styles["CoverMeta"],
        )
    )
    story.append(
        _p(
            "The figures in this teaching note are rounded illustrations for a software demo. "
            "They are not a substitute for cancer-registry statistics or clinical guidelines.",
            styles["CoverMeta"],
        )
    )
    story.append(Spacer(1, 8))

    story.append(_p("1. Introduction", styles["Heading1Custom"]))
    story.append(
        _p(
            "Brain tumors are a heterogeneous group of neoplasms that arise from the brain "
            "parenchyma, the meninges, or adjacent structures such as the pituitary gland. "
            "In routine clinical imaging and in many public MRI datasets used for research, "
            "four labels appear again and again: glioma, meningioma, pituitary tumor, and "
            "no tumor (a scan without a visible neoplasm).",
            styles["BodyJust"],
        )
    )
    story.append(
        _p(
            "A glioma is a tumor that originates from glial cells &mdash; astrocytes, "
            "oligodendrocytes, or ependymal cells. High-grade gliomas, especially "
            "glioblastoma, are aggressive, infiltrate surrounding white matter, and are "
            "difficult to resect completely. Low-grade gliomas grow more slowly but may "
            "still undergo malignant transformation over time.",
            styles["BodyJust"],
        )
    )
    story.append(
        _p(
            "A meningioma arises from the meninges, the membranous layers that cover the "
            "brain and spinal cord. Most meningiomas are extra-axial and histologically "
            "benign (WHO grade 1), and they often compress rather than infiltrate the brain.",
            styles["BodyJust"],
        )
    )
    story.append(
        _p(
            "A pituitary tumor, most commonly a pituitary adenoma, develops in the sellar "
            "region. Even when benign, these tumors can affect vision by compressing the "
            "optic chiasm and can disrupt hormone production.",
            styles["BodyJust"],
        )
    )
    story.append(
        _p(
            "The phrase &quot;no tumor&quot; in a classification setting simply means that "
            "the MRI study does not show a space-occupying neoplasm of these types. It is "
            "not a clinical diagnosis by itself.",
            styles["BodyJust"],
        )
    )

    story.append(_p("2. MRI sequences used to characterise brain tumors", styles["Heading1Custom"]))
    story.append(
        _p(
            "Magnetic resonance imaging is the primary non-invasive modality for "
            "characterising brain tumors before surgery. Different pulse sequences "
            "highlight different tissue properties, and radiologists usually read them together.",
            styles["BodyJust"],
        )
    )
    story.append(
        _p(
            "T1-weighted images depict anatomy with high spatial detail. Cerebrospinal fluid "
            "appears dark, and gray matter is darker than white matter. After gadolinium "
            "contrast is given, a T1-contrast (T1C) sequence shows regions where the "
            "blood&ndash;brain barrier is disrupted. Many high-grade gliomas and meningiomas "
            "enhance, while some low-grade gliomas enhance weakly or not at all.",
            styles["BodyJust"],
        )
    )
    story.append(
        _p(
            "T2-weighted images show fluid as bright signal. Edema and many tumors appear "
            "hyperintense on T2, but the boundary between tumor and edema can be hard to "
            "judge on T2 alone.",
            styles["BodyJust"],
        )
    )
    story.append(
        _p(
            "FLAIR (Fluid-Attenuated Inversion Recovery) suppresses the bright signal of "
            "cerebrospinal fluid while leaving other T2-hyperintense tissue visible. FLAIR "
            "is particularly useful for detecting edema and lesions that lie next to the "
            "ventricles or the cerebral sulci, where bright cerebrospinal fluid would "
            "otherwise conceal the finding.",
            styles["BodyJust"],
        )
    )
    story.append(
        _p(
            "A typical research MRI study for tumor classification therefore includes at "
            "least T1, T1-contrast, T2, and FLAIR, although many open datasets also provide "
            "only a single sequence or a single two-dimensional slice.",
            styles["BodyJust"],
        )
    )

    story.append(_p("3. Clinical features at a glance", styles["Heading1Custom"]))
    story.append(
        _p(
            "Age and location are useful priors when reading a scan, but they never replace "
            "formal radiological and histopathological assessment. For example, a "
            "contrast-enhancing extra-axial mass over the convexity in a middle-aged adult "
            "is more often a meningioma than a glioma, whereas an irregular, rim-enhancing "
            "intra-axial mass with substantial surrounding edema raises concern for glioblastoma.",
            styles["BodyJust"],
        )
    )

    def c(text: str, head: bool = False) -> Paragraph:
        return Paragraph(text, header if head else cell)

    table_data = [
        [
            c("Tumor type", True),
            c("Typical adult age range", True),
            c("Usual location", True),
            c("Approx. share of primary brain tumors", True),
            c("Preferred first-line treatment (general)", True),
        ],
        [
            c("Meningioma"),
            c("40&ndash;70 years"),
            c("Meninges, often parasagittal or sphenoid wing"),
            c("about 40%"),
            c("Surgical resection when indicated"),
        ],
        [
            c("Glioblastoma (high-grade glioma)"),
            c("45&ndash;70 years"),
            c("Cerebral hemispheres"),
            c("about 15%"),
            c("Maximal safe resection, then radiotherapy with temozolomide"),
        ],
        [
            c("Pituitary adenoma"),
            c("30&ndash;60 years"),
            c("Sellar / suprasellar region"),
            c("about 15%"),
            c("Transsphenoidal surgery or medical therapy, depending on type"),
        ],
        [
            c("Low-grade glioma"),
            c("20&ndash;45 years"),
            c("Cerebral hemispheres, often frontal or temporal"),
            c("under 15%"),
            c("Surgery, with radiotherapy or chemotherapy in selected cases"),
        ],
    ]
    table = Table(table_data, colWidths=[1.15 * inch, 1.05 * inch, 1.35 * inch, 1.15 * inch, 1.55 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f4a6e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#eef3f8")),
                ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#eef3f8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#8899aa")),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)
    story.append(
        _p(
            "Table 1. Illustrative clinical features of common intracranial tumors. "
            "Shares and age ranges are rounded teaching estimates, not registry data.",
            styles["Caption"],
        )
    )
    story.append(
        _p(
            "Two treatment details from the table are worth restating in prose because they "
            "are easy to miss when a table is copied poorly. Glioblastoma is generally managed "
            "with maximal safe resection followed by radiotherapy with temozolomide. "
            "A hormone-secreting pituitary adenoma may be treated medically rather than "
            "with immediate surgery, depending on the secretory type.",
            styles["BodyJust"],
        )
    )

    story.append(_p("4. Survival, automated reading, and limits", styles["Heading1Custom"]))
    story.append(
        _p(
            "Survival varies sharply by histology and grade. Glioblastoma still has a "
            "markedly poorer prognosis than WHO grade 1 meningioma or most pituitary "
            "adenomas, even with current standard therapy. Figure 1 below is a teaching "
            "graphic of illustrative five-year survival by tumor type. The exact percentages "
            "are shown only in the figure and are not restated in this accompanying text. "
            "These teaching figures should not be quoted as official registry statistics.",
            styles["BodyJust"],
        )
    )
    story.append(
        KeepTogether(
            [
                RLImage(str(figure_path), width=6.3 * inch, height=3.675 * inch),
                _p(
                    "Figure 1. Illustrative five-year survival by tumor type. Values appear "
                    "only in the graphic itself.",
                    styles["Caption"],
                ),
            ]
        )
    )
    story.append(
        _p(
            "Automated systems that classify MRI slices into glioma, meningioma, pituitary "
            "tumor, or no tumor can be useful as research prototypes. They should not be "
            "treated as clinical diagnostic devices. Models trained on two-dimensional slices "
            "miss volumetric context, and models trained on a single public dataset may not "
            "generalise to scanners, protocols, or populations they have never seen. When "
            "such a system is used for teaching or research, an explanation method such as a "
            "Grad-CAM heatmap can show which region of the slice influenced the prediction, "
            "but a heatmap is not a segmentation of the tumor.",
            styles["BodyJust"],
        )
    )
    story.append(
        _p(
            "The most common failure modes in automated MRI classification include confusing "
            "glioma with meningioma when the mass sits near the cortical surface, missing "
            "small pituitary lesions on slices that barely include the sella, and over-calling "
            "tumor on scans that actually show post-treatment change or inflammation.",
            styles["BodyJust"],
        )
    )

    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Times-Roman", 8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(
            0.9 * inch,
            0.45 * inch,
            "Rikdeb Mandal, 2026. CC BY 4.0. Teaching sample for pdfqa; not clinical advice.",
        )
        canvas.drawRightString(letter[0] - 0.9 * inch, 0.45 * inch, f"Page {doc_.page}")
        canvas.restoreState()

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return pdf_path


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fig = render_figure(Path(tmp) / "figure1_survival.png")
        pdf = build_pdf(PDF_PATH, fig)
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()

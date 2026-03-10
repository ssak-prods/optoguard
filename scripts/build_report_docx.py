from pathlib import Path

from docx import Document
from docx.shared import Inches


ROOT = Path(r"d:\optoguard")
TEXT_PATH = ROOT / "technical-report.txt"
PLOTS = [
    ("Figure 1. Uncertainty vs. lighting condition", ROOT / "plotsfolder" / "uncVLightV2.png"),
    ("Figure 2. Uncertainty vs. occlusion level", ROOT / "plotsfolder" / "uncVoccV2.png"),
    ("Figure 3. Latency by condition and hardware", ROOT / "plotsfolder" / "latV2.png"),
    ("Figure 4. Development timeline (Oct 2025–Mar 2026)", ROOT / "plotsfolder" / "timeline.png"),
]

OUT_DIR = ROOT / "paper"
OUT_PATH = OUT_DIR / "OptoGuard_Technical_Report.docx"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = TEXT_PATH.read_text(encoding="utf-8")
    lines = [ln.rstrip() for ln in raw.splitlines()]

    doc = Document()

    # Manual contents page: simple static list of sections (no automatic page numbers).
    doc.add_heading("Contents", level=1)
    toc_entries = [
        "1. Introduction",
        "2. Related Work",
        "3. Methodology",
        "4. Experimental Setup",
        "5. Results",
        "   5.1 Uncertainty Under Occlusion",
        "   5.2 Uncertainty Under Lighting Variation",
        "   5.3 Latency on Laptop and Raspberry Pi 5",
        "   5.4 Development Timeline",
        "6. Discussion and Limitations",
        "7. Conclusion and Future Work",
    ]
    for entry in toc_entries:
        doc.add_paragraph(entry)

    doc.add_page_break()

    # Manual list of figures.
    doc.add_heading("List of Figures", level=1)
    lof_entries = [
        "Figure 1. Uncertainty vs. lighting condition",
        "Figure 2. Uncertainty vs. occlusion level",
        "Figure 3. Latency by condition and hardware",
        "Figure 4. Development timeline (Oct 2025–Mar 2026)",
    ]
    for entry in lof_entries:
        doc.add_paragraph(entry)

    doc.add_page_break()

    # Main title and body.
    doc.add_heading(
        "OptoGuard: Uncertainty-Aware Object Detection under Distribution Shift on Edge Hardware",
        level=0,
    )
    doc.add_paragraph(
        "Technical report on Monte Carlo Dropout-based uncertainty analysis for YOLOv8n under distribution shift."
    )
    doc.add_paragraph("")

    for ln in lines:
        s = ln.strip()
        if not s or set(s) == {"_"}:
            continue

        # Top-level headings such as "1. Introduction"
        if len(s) > 3 and s[0].isdigit() and s[1:3] == ". ":
            doc.add_heading(s, level=1)
            continue

        # Sub-headings such as "3.1 Base Detector ..."
        parts = s.split(" ", 1)
        if parts and "." in parts[0] and parts[0].replace(".", "").isdigit() and len(parts[0]) >= 3:
            doc.add_heading(s, level=2)
            continue

        doc.add_paragraph(s)

    # Figures section (static captions).
    doc.add_page_break()
    doc.add_heading("Figures", level=1)
    for caption, path in PLOTS:
        if path.exists():
            cap_para = doc.add_paragraph(caption)
            doc.add_picture(str(path), width=Inches(6.5))
            doc.add_paragraph("")

    doc.save(str(OUT_PATH))
    print(OUT_PATH)


if __name__ == "__main__":
    main()


"""Export chooser: organized typeset output or faithful input layout."""
from __future__ import annotations

from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                             QFormLayout, QLabel, QVBoxLayout)

CHOICES = [
    ("PDF · Organized (reflowed)", "pdf", "organized",
     "A clean, typeset PDF. Headings, equations, tables and diagrams are "
     "reflowed into a readable single-column document."),
    ("PDF · Same layout as input", "pdf", "faithful",
     "Keeps every page exactly as it looked in the scan and adds an invisible "
     "searchable text layer from the VLM transcription."),
    ("LaTeX · Annotated (SSOT)", "tex", "organized",
     "The full annotated LaTeX source with %PX block markers."),
    ("LaTeX · Clean", "clean-tex", "organized",
     "LaTeX without provenance markers, ready to compile."),
    ("DOCX · Word document", "docx", "organized",
     "Editable Word document with headings, paragraphs and tables."),
    ("HTML · Web page", "html", "organized",
     "Self-contained HTML preview of the document."),
    ("Markdown", "md", "organized",
     "Portable Markdown with equations and tables."),
    ("JSON · Structured blocks", "json", "organized",
     "Every page, block, bbox and kind for downstream tooling."),
]


class ExportDialog(QDialog):
    """Pick a format and, for PDF, the layout contract."""

    def __init__(self, parent=None, *, title: str = "Export document"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(520)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 14)
        root.setSpacing(10)

        heading = QLabel("Choose the output that fits the job")
        heading.setObjectName("pageTitle")
        note = QLabel(
            "PageX used the VLM to preserve the page structure. "
            "Keep that structure with a faithful PDF, or let PageX clean it "
            "up into an organized document.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        root.addWidget(heading)
        root.addWidget(note)

        form = QFormLayout()
        self.combo = QComboBox()
        for label, fmt, layout, _desc in CHOICES:
            self.combo.addItem(label, (fmt, layout))
        self.combo.currentIndexChanged.connect(self._update_description)
        form.addRow("Format", self.combo)
        root.addLayout(form)

        self.description = QLabel("")
        self.description.setObjectName("muted")
        self.description.setWordWrap(True)
        self.description.setMinimumHeight(58)
        root.addWidget(self.description)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.Ok)
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setText("Continue")
        ok.setProperty("primary", True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._update_description()

    def _update_description(self) -> None:
        idx = self.combo.currentIndex()
        if 0 <= idx < len(CHOICES):
            self.description.setText(CHOICES[idx][3])

    def selection(self) -> tuple[str, str]:
        """Return ``(format, pdf_layout)``."""
        data = self.combo.currentData()
        return (str(data[0]), str(data[1])) if data else ("pdf", "organized")

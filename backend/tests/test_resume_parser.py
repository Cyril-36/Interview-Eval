"""Tests for resume PDF parsing and skill extraction."""
import io

import pytest
from pypdf import PdfWriter
from reportlab.pdfgen import canvas

from app.services.question_generator import _format_resume_block
from app.services.resume_parser import (
    ResumeParseError,
    extract_skills,
    parse_pdf,
)


def _make_pdf(text: str) -> bytes:
    """Render a one-page PDF with the given text (for test fixtures)."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    width, height = c._pagesize
    y = height - 50
    for line in text.split("\n"):
        c.drawString(50, y, line)
        y -= 14
        if y < 50:
            c.showPage()
            y = height - 50
    c.save()
    return buf.getvalue()


def _make_empty_pdf() -> bytes:
    """Generate a valid PDF with no extractable text."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class TestExtractSkills:
    def test_empty_text_returns_empty_list(self):
        assert extract_skills("") == []

    def test_vocabulary_matches_picked_up(self):
        text = (
            "Senior Software Engineer with deep expertise in Python, FastAPI,"
            " React, PostgreSQL, Docker, and Kubernetes. Built RAG pipelines"
            " on AWS using PyTorch."
        )
        skills = extract_skills(text, top_n=12)
        assert "python" in skills
        assert "fastapi" in skills
        assert "react" in skills
        assert "postgresql" in skills
        assert "docker" in skills
        assert "kubernetes" in skills
        assert "aws" in skills
        assert "pytorch" in skills
        assert "rag" in skills

    def test_skills_respect_word_boundaries(self):
        # 'rusty' should not match 'rust'
        skills = extract_skills("I am a rusty cook with goofy go habits.")
        assert "rust" not in skills

    def test_top_n_caps_output(self):
        text = (
            "Python Java JavaScript Go Rust Ruby PHP Swift Kotlin Scala SQL"
            " React Angular Vue Docker AWS"
        )
        assert len(extract_skills(text, top_n=5)) == 5


class TestParsePDF:
    def test_rejects_empty_bytes(self):
        with pytest.raises(ResumeParseError):
            parse_pdf(b"")

    def test_rejects_non_pdf_bytes(self):
        with pytest.raises(ResumeParseError):
            parse_pdf(b"this is not a pdf")

    def test_rejects_when_over_size_limit(self):
        with pytest.raises(ResumeParseError):
            parse_pdf(b"x" * 100, max_bytes=50)

    def test_extracts_text_and_skills_from_real_pdf(self):
        pdf_bytes = _make_pdf(
            "Jane Doe — Senior Software Engineer\n"
            "Skills: Python, FastAPI, React, Docker, AWS, PostgreSQL\n"
            "Experience: Built distributed systems with Kafka and Redis."
        )
        parsed = parse_pdf(pdf_bytes)
        assert "python" in parsed.skills
        assert "fastapi" in parsed.skills
        assert "react" in parsed.skills
        assert parsed.page_count == 1
        assert parsed.truncated is False
        assert "Jane Doe" in parsed.text or "Jane" in parsed.text

    def test_truncates_when_text_exceeds_limit(self):
        long_text = "Python expert. " * 1000
        pdf_bytes = _make_pdf(long_text)
        parsed = parse_pdf(pdf_bytes, max_chars=200)
        assert parsed.truncated is True
        assert len(parsed.text) <= 200

    def test_raises_when_no_extractable_text(self):
        with pytest.raises(ResumeParseError):
            parse_pdf(_make_empty_pdf())


class TestFormatResumeBlock:
    def test_empty_when_no_context_or_skills(self):
        assert _format_resume_block(None, None) == ""
        assert _format_resume_block("", []) == ""

    def test_includes_skills_and_context_when_provided(self):
        block = _format_resume_block(
            "10 years building distributed Python services on AWS.",
            ["python", "aws", "kubernetes"],
        )
        assert "python" in block
        assert "aws" in block
        assert "kubernetes" in block
        assert "10 years" in block
        assert "tailor questions" in block.lower()

    def test_caps_resume_summary_length(self):
        block = _format_resume_block("x" * 5000, [])
        assert "x" * 1500 in block
        assert "x" * 1501 not in block

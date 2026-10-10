import os
import sys

# 1. Environment configuration for Surya / Marker 2.x
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["SURYA_INFERENCE_BACKEND"] = "llamacpp"
os.environ["MARKER_NUM_THREADS"] = "1"
os.environ["PDFTEXT_WORKERS"] = "1"

import json
from pathlib import Path
from PIL import Image

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered


def json_serializer(obj):
    """Custom serializer for objects that default json cannot handle."""
    if isinstance(obj, Image.Image):
        return f"<PIL.Image size={obj.size} mode={obj.mode}>"
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def walk(block, out):
    t = block.get("block_type")
    if t in ("Table", "Figure", "Picture", "Caption", "TableGroup", "FigureGroup"):
        try:
            page = int(block["id"].split("/")[2])
        except (KeyError, IndexError, ValueError):
            page = block.get("page", 0)

        out.append({
            "id": block.get("id"),
            "type": t,
            "page": page,
            "html": block.get("html"),
            "polygon": block.get("polygon"),
        })

    for child in block.get("children") or []:
        walk(child, out)


def read_pdf(pdf_path: str, out_dir: str):
    pdf = Path(pdf_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not pdf.is_file():
        raise FileNotFoundError(f"Input PDF not found: {pdf}")

    print("[1/4] Configuring converter settings...")
    cfg = {
        "output_format": "markdown",
        "mode": "balanced",
        "disable_ocr": False,
        "disable_multiprocessing": True,
        "pdftext_workers": 1,
    }
    parser = ConfigParser(cfg)

    print("[2/4] Loading models with llamacpp backend...")
    models = create_model_dict()

    conv = PdfConverter(
        config=parser.generate_config_dict(),
        artifact_dict=models,
        processor_list=parser.get_processors(),
        renderer=parser.get_renderer(),
        llm_service=parser.get_llm_service()
    )

    print(f"[3/4] Converting {pdf.name}...")
    rendered = conv(str(pdf))

    print("[4/4] Writing output files...")
    text, _, images = text_from_rendered(rendered)

    # 1. Save Markdown
    (out / "paper.md").write_text(text, encoding="utf-8")
    print(f"      Saved paper.md ({len(text)} characters)")

    # 2. Save Extracted Images
    for name, img in images.items():
        img.save(out / name)
    print(f"      Extracted {len(images)} images")

    # 3. Dump JSON with safe serializer
    rendered_dict = rendered.model_dump()
    blocks = []
    for page in rendered_dict.get("children", []):
        walk(page, blocks)

    (out / "blocks.json").write_text(
        json.dumps(blocks, indent=2, default=json_serializer), 
        encoding="utf-8"
    )
    (out / "full_rendered.json").write_text(
        json.dumps(rendered_dict, indent=2, default=json_serializer), 
        encoding="utf-8"
    )

    print(f"Success! All files cleanly exported to: {out}")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python marker_pdf.py <input_pdf_path> <output_dir>")
        sys.exit(1)

    read_pdf(sys.argv[1], sys.argv[2])
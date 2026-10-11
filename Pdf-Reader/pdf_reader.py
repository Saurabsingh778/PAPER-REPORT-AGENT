import sys
import json
from pathlib import Path

# Run as a plain script ("Pdf-Reader" is not an importable package name), so put the
# project root on sys.path to make the shared Logger package importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docling.document_converter import DocumentConverter
from Logger import get_phase, link_artifact, phase, start_run

def read_pdf(pdf_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading and parsing {pdf_path} with Docling...")
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    
    # 1. Export Clean Markdown
    markdown_content = result.document.export_to_markdown()
    (out / "paper.md").write_text(markdown_content, encoding="utf-8")
    print(f"Saved Markdown to {out / 'paper.md'}")
    
    # 2. Export full document structure as JSON
    doc_dict = result.document.export_to_dict()
    (out / "docling_doc.json").write_text(json.dumps(doc_dict, indent=2), encoding="utf-8")
    
    # 3. Extract Tables directly to Markdown or HTML
    tables_dir = out / "tables"
    tables_dir.mkdir(exist_ok=True)
    for i, table in enumerate(result.document.tables):
        table_df = table.export_to_dataframe()
        table_df.to_csv(tables_dir / f"table_{i+1}.csv", index=False)
    
    print(f"Extracted {len(result.document.tables)} tables to {tables_dir}")
    print("Parsing completed successfully!")

    get_phase().info("pdf parsed", pdf=str(pdf_path), markdown_chars=len(markdown_content),
                     tables=len(result.document.tables), output_dir=str(out))
    # Lets the extraction run find this run as its upstream.
    link_artifact(out / "paper.md", "paper_markdown")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python dockling_reader.py <input_pdf_path> <output_dir>")
        sys.exit(1)
    with start_run("phase0_pdf_read", meta={"pdf": sys.argv[1], "out_dir": sys.argv[2], "engine": "docling"}):
        with phase("phase0_pdf_read", "convert the PDF to markdown, tables and document JSON"):
            read_pdf(sys.argv[1], sys.argv[2])
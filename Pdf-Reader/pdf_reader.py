import sys
import json
from pathlib import Path
from docling.document_converter import DocumentConverter

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

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python dockling_reader.py <input_pdf_path> <output_dir>")
        sys.exit(1)
    read_pdf(sys.argv[1], sys.argv[2])
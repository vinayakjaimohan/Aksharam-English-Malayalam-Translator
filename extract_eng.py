import fitz

pdf_path = "novel.pdf"
output_txt = "english_full.txt"

doc = fitz.open(pdf_path)

with open(output_txt, "w", encoding="utf-8") as f:
    for page in doc:
        text = page.get_text("text")
        f.write(text + "\n")

doc.close()

print("English text extracted successfully")

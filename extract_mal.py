"""
Malayalam OCR using Tesseract (Windows Ready Version)
"""

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os


class MalayalamOCR:

    def __init__(self, tesseract_path=None):
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        self.verify_malayalam_support()


    def verify_malayalam_support(self):
        try:
            langs = pytesseract.get_languages(config='')
            if 'mal' not in langs:
                raise RuntimeError("Malayalam language pack not found.")
            print("✓ Malayalam language support verified")
        except Exception as e:
            print("Warning:", e)


    def pdf_to_images(self, pdf_path, dpi=400):
        print(f"Converting PDF to images (DPI: {dpi})...")

        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            poppler_path=r"C:\Users\Acer\Downloads\Release-25.12.0-0\poppler\Library\bin"
        )

        print(f"✓ Converted {len(images)} pages to images")
        return images


    def image_to_text(self, image, lang='mal', config='--psm 3'):
        return pytesseract.image_to_string(image, lang=lang, config=config)


    def process_pdf(self, pdf_path, output_file='malayalam_novel.txt'):
        images = self.pdf_to_images(pdf_path)

        all_text = []
        for i, image in enumerate(images, 1):
            print(f"Processing page {i}/{len(images)}...")
            text = self.image_to_text(image)
            all_text.append(text)

        combined_text = "\n\n".join(all_text)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(combined_text)

        print("✅ OCR COMPLETE")
        print(f"Output saved to: {output_file}")

        return combined_text



# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------
def main():

    PDF_PATH = "malayalam_novel.pdf"
    OUTPUT_FILE = "malayalam_novel.txt"

    TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    ocr = MalayalamOCR(tesseract_path=TESSERACT_PATH)

    text = ocr.process_pdf(
        pdf_path=PDF_PATH,
        output_file=OUTPUT_FILE
    )

    print("\nPreview of extracted text:\n")
    print(text[:500])


if __name__ == "__main__":
    main()

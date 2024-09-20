import re
from pdf2image import convert_from_bytes
import pytesseract
from PyPDF2 import PdfReader

def get_pdf_text(pdf, page_range: tuple):
    text = ""
    pdf_reader = PdfReader(pdf)

    images = convert_from_bytes(
        pdf.getvalue(), first_page=page_range[0], last_page=page_range[1]
    )
    for page_num, image in enumerate(images, start=page_range[0]):
        image_text = pytesseract.image_to_string(image)
        if image_text.strip():
            text += f"PAGE {page_num}: {image_text}"

    text = re.sub(r"[\s+\n]", " ", text)
    text = re.sub(r"\f", "", text)

    return text


def page_count(pdf):
    pdf_reader = PdfReader(pdf)
    return len(pdf_reader.pages)
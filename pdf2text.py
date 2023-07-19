from pdf2image import convert_from_path
import pytesseract
import os
import easyocr


class PDFtoTXTConverter:
    def __init__(self, pdf_path, model):
        self.pdf_path = pdf_path
        self.model = model
        self.reader = easyocr.Reader(['en'])

    def convert_to_images(self):
        images = convert_from_path(self.pdf_path)
        return images

    def perform_ocr(self, images):
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img)
        return text

    def perform_easy_ocr(self, images):
        text = ""
        for img in images:
            result = self.reader.readtext(img)
            for line in result:
                text += line[1] + " "
        return text

    def save_as_text(self, text, output_path):
        with open(output_path, 'w') as txt_file:
            txt_file.write(text)

        print(f"Text saved as {output_path}")

    def convert_to_text(self, output_path):
        images = self.convert_to_images()
        if self.model == "easyocr":
            text = self.perform_easy_ocr(images)
        else:
            text = self.perform_ocr(images)
        self.save_as_text(text, output_path)


if __name__ == "__main__":
    for pdf_file in os.listdir("visa_data"):
        pdf_file = f"visa_data/{pdf_file}"
        output_path = f"{pdf_file[:-4]}_2.txt"
        converter = PDFtoTXTConverter(pdf_file, model="easyocr")
        converter.convert_to_text(output_path)

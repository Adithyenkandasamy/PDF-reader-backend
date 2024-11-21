import PyPDF2
import os

def extract_pdf_text(pdf_path):
    # Check if the file exists
    if not os.path.exists(pdf_path):
        print(f"Error: The file '{pdf_path}' does not exist.")
        return None

    try:
        # Open the PDF file
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            # Loop through all the pages and extract text
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"

            return text

    except Exception as e:
        print(f"Error processing the PDF: {str(e)}")
        return None

def main():
    # Specify the location of your PDF file
    pdf_path = 'path_to_your_pdf_file.pdf'

    # Extract text from PDF
    extracted_text = extract_pdf_text(pdf_path)

    # If text is extracted, print it
    if extracted_text:
        print("Extracted Text from PDF:")
        print(extracted_text)
    else:
        print("No text could be extracted.")

if __name__ == '__main__':
    main()

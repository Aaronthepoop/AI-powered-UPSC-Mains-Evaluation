import boto3
from PIL import Image
import pdf2image

def extract_text_from_pdf(pdf_path):
    # Convert PDF to images
    images = pdf2image.convert_from_path(pdf_path)
    text = ""
    for image in images:
        # Save image and pass to Textract
        text += extract_text_from_image(image)
    return text

def extract_text_from_image(image):
    # Use Textract for image
    textract_client = boto3.client('textract')
    image_bytes = image.tobytes()  # Convert to byte format
    response = textract_client.detect_document_text(Document={'Bytes': image_bytes})
    text = ""
    for item in response['Blocks']:
        if item['BlockType'] == 'LINE':
            text += item['Text'] + "\n"
    return text

# Example usage
if __name__ == "__main__":
    pdf_text = extract_text_from_pdf('student_submission.pdf')
    print(pdf_text)

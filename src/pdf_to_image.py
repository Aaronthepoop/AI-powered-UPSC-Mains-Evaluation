from pdf2image import convert_from_path

def convert_pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    return images

# Example usage:
images = convert_pdf_to_images('student_submission.pdf')
for i, image in enumerate(images):
    image.save(f'page_{i + 1}.png', 'PNG')

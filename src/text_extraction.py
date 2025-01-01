import boto3
from time import sleep

def extract_text_from_pdf(file_path):
    client = boto3.client('textract')
    
    # Open the PDF and send it to Textract
    with open(file_path, 'rb') as file:
        response = client.start_document_text_detection(
            Document={'Bytes': file.read()}
        )
    
    job_id = response['JobId']
    
    # Wait for the job to complete
    while True:
        result = client.get_document_text_detection(JobId=job_id)
        status = result['JobStatus']
        if status in ['SUCCEEDED', 'FAILED']:
            break
        sleep(5)
    
    if status == 'SUCCEEDED':
        text = ''
        for item in result['Blocks']:
            if item['BlockType'] == 'LINE':
                text += item['Text'] + '\n'
        return text
    else:
        raise Exception("Textract job failed")

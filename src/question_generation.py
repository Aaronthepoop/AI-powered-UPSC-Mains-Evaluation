import openai
import os

# Load API key from .env
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_questions(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",  # Or choose the right model
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Example usage
if __name__ == "__main__":
    prompt = "Generate a question based on the UPSC Mains syllabus for General Studies."
    print(generate_questions(prompt))

from transformers import BertTokenizer, BertForSequenceClassification
import torch

def grade_answer(answer):
    model_name = "bert-base-uncased"
    model = BertForSequenceClassification.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    inputs = tokenizer(answer, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    # Grading logic based on logits, e.g., assigning a grade
    grade = "A" if torch.argmax(logits) == 0 else "B"
    return grade, "Well-written answer."

# Example usage
if __name__ == "__main__":
    answer = "This is a well-written answer."
    grade, feedback = grade_answer(answer)
    print(f"Grade: {grade}, Feedback: {feedback}")

from transformers import T5ForConditionalGeneration, T5Tokenizer

def enhance_answer(answer):
    model_name = "t5-base"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    input_text = f"Improve the following answer: {answer}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids)

    enhanced_answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return enhanced_answer

# Example usage
if __name__ == "__main__":
    original_answer = "This is a basic answer."
    print(enhance_answer(original_answer))

import json
from transformers import Trainer, TrainingArguments, BertForSequenceClassification, BertTokenizer

def fine_tune_model(dataset_path):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Prepare dataset (adjust according to your dataset format)
    inputs = tokenizer([item['question'] for item in dataset], padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor([item['grade'] for item in dataset])

    # Fine-tuning logic here
    training_args = TrainingArguments(output_dir='./results', num_train_epochs=1, per_device_train_batch_size=8)
    trainer = Trainer(model=model, args=training_args, train_dataset=inputs, eval_dataset=labels)
    trainer.train()

if __name__ == "__main__":
    fine_tune_model("data/question_answer_dataset.json")

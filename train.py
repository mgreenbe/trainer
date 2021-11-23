import logging
import sys
import wandb
from datasets import Dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

PRETRAINED_MODEL_NAME_OR_PATH = "bertlet"

if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH)

    imdb = Dataset.from_json("imdb.json", keep_in_memory=True)

    def mapper(x):
        return tokenizer(x["text"], max_length=384, truncation=True, padding="max_length")

    train_dataset = imdb.select(range(200)).map(
        mapper, remove_columns=["text"])
    eval_dataset = imdb.select(range(200, 250)).map(
        mapper, remove_columns=["text"])

    accuracy = load_metric("accuracy")

    def compute_metrics(outputs):
        predictions = outputs.predictions.argmax(axis=1)
        references = outputs.label_ids
        return accuracy.compute(predictions=predictions, references=references)

    args = TrainingArguments(
        "imdb-blah-blah",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        report_to="wandb",
        save_strategy="steps",
        save_steps=3,
        evaluation_strategy="steps",
        eval_steps=3
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()

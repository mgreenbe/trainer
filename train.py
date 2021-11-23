import logging
import sys
import wandb
from dotenv import load_dotenv
from argparse import ArgumentParser
from datasets import Dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

PRETRAINED_MODEL_NAME_OR_PATH = "bertlet"

if __name__ == "__main__":
    load_dotenv()
    logger = logging.getLogger(__name__)

    parser = ArgumentParser()

    parser.add_argument("--save_strategy", type=str, default="no")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--report_to", type=str, default="none")

    args, _ = parser.parse_known_args()

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
        "models/imdb-blah-blah",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=2,
        report_to=args.report_to,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
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

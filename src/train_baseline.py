import os
import argparse
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Trainer
from evaluate import load as load_metric
from src.utils import set_seed, load_json, save_json
from src.data_preprocessing import build_preprocess_fn

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Path to JSON config")
    ap.add_argument("--model_name", type=str, default="facebook/bart-large-cnn")
    ap.add_argument("--output_dir", type=str, default="results/baseline")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=4)
    ap.add_argument("--max_src_len", type=int, default=1024)
    ap.add_argument("--max_tgt_len", type=int, default=128)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--eval_steps", type=int, default=1000)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    if args.config:
        cfg = load_json(args.config)
        for k, v in cfg.items():
            setattr(args, k, v)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset("cnn_dailymail", "3.0.0")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    preprocess_fn = build_preprocess_fn(tokenizer, args.max_src_len, args.max_tgt_len)
    tokenized = dataset.map(preprocess_fn, batched=True, remove_columns=dataset["train"].column_names)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, attn_implementation="eager")
    model.config.use_cache = False     # lowers memory during training
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    rouge = load_metric("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = [[(tok if tok != -100 else tokenizer.pad_token_id) for tok in label] for label in labels]
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=pred_str, references=label_str, use_stemmer=True)
        # return {k: round(v.mid.fmeasure * 100, 2) for k, v in result.items()}
        norm = {}
        for k, v in result.items():
            try:
                val = float(v.mid.fmeasure)
            except AttributeError:
                val = float(v)
            norm[k] = round(val * 100, 2)
        return norm

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        do_eval=True,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
    gradient_checkpointing=True,
        predict_with_generate=True,
        generation_max_length=args.max_tgt_len,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    save_json(metrics, os.path.join(args.output_dir, "metrics.json"))

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
# train_phobert_manual.py
# Fine-tune PhoBERT for hate speech detection (multi-class)
# Train on train.csv, validate on dev.csv, test once on test.csv
#
# pip install -U transformers accelerate torch scikit-learn pandas numpy

import os
import json
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)

# ===================== CONFIG =====================
# TRAIN_PATH = "/home/uit2023/LuuTru/Thuchd/cs221/CS221_NLP_SA/vihsd/train.csv"
# DEV_PATH   = "/home/uit2023/LuuTru/Thuchd/cs221/CS221_NLP_SA/vihsd/dev.csv"
# TEST_PATH  = "/home/uit2023/LuuTru/Thuchd/cs221/CS221_NLP_SA/vihsd/test.csv"

TRAIN_PATH = "/home/uit2023/LuuTru/Thuchd/cs221/CS221_NLP_SA/UIT-ViHSD-preprocessed/train.csv"
DEV_PATH   = "/home/uit2023/LuuTru/Thuchd/cs221/CS221_NLP_SA/UIT-ViHSD-preprocessed/dev.csv"
TEST_PATH  = "/home/uit2023/LuuTru/Thuchd/cs221/CS221_NLP_SA/UIT-ViHSD-preprocessed/test.csv"

TEXT_COL  = "free_text"
LABEL_COL = "label_id"

MODEL_NAME = "vinai/phobert-base"   # hoặc "vinai/phobert-large"
OUTPUT_DIR = "./models/phobert_viHSD_manual"
SAVE_FINAL_DIR = os.path.join(OUTPUT_DIR, "final_model")
SAVE_INFO_PATH = os.path.join(OUTPUT_DIR, "final_info.json")

RANDOM_STATE = 42

# ===================== HYPERPARAMS (MANUAL) =====================
HP = {
    "learning_rate": 4e-5,
    "batch_size": 64,          # per GPU
    "epochs": 4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "max_length": 128,
    "grad_accum": 1,
}

# Early stopping (nếu muốn dùng dev để dừng sớm)
USE_EARLY_STOP = True
EARLY_STOP_PATIENCE = 2

# ===================== UTIL =====================
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=[TEXT_COL, LABEL_COL]).copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df

def overlap_count(a_texts, b_texts) -> int:
    return len(set(a_texts) & set(b_texts))

def eval_and_print(name, y_true, y_pred, label_names=None):
    print(f"\n===== {name} =====")
    print("f1_macro:", f1_score(y_true, y_pred, average="macro"))
    print("acc     :", accuracy_score(y_true, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nReport:\n", classification_report(y_true, y_pred, digits=4, target_names=label_names))

def make_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "f1_macro": f1_score(labels, preds, average="macro"),
            "accuracy": accuracy_score(labels, preds),
        }
    return compute_metrics

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def _training_args_eval_key() -> str:
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        return "eval_strategy"
    return "evaluation_strategy"

def make_training_args(**kwargs) -> TrainingArguments:
    eval_key = _training_args_eval_key()
    if "evaluation_strategy" in kwargs and eval_key != "evaluation_strategy":
        kwargs[eval_key] = kwargs.pop("evaluation_strategy")
    if "eval_strategy" in kwargs and eval_key != "eval_strategy":
        kwargs[eval_key] = kwargs.pop("eval_strategy")
    return TrainingArguments(**kwargs)

def make_trainer(**kwargs) -> Trainer:
    tokenizer = kwargs.pop("tokenizer", None)
    if tokenizer is None:
        return Trainer(**kwargs)
    try:
        return Trainer(processing_class=tokenizer, **kwargs)
    except TypeError:
        return Trainer(tokenizer=tokenizer, **kwargs)

def remap_labels(y_train, y_dev, y_test):
    unique = sorted(set(y_train) | set(y_dev) | set(y_test))
    label2id = {lab: i for i, lab in enumerate(unique)}
    id2label = {i: str(lab) for lab, i in label2id.items()}
    y_train_m = [label2id[x] for x in y_train]
    y_dev_m   = [label2id[x] for x in y_dev]
    y_test_m  = [label2id[x] for x in y_test]
    return y_train_m, y_dev_m, y_test_m, label2id, id2label

# ===================== MAIN =====================
def main():
    set_seed(RANDOM_STATE)

    df_train = load_df(TRAIN_PATH)
    df_dev   = load_df(DEV_PATH)
    df_test  = load_df(TEST_PATH)

    X_train = df_train[TEXT_COL].tolist()
    y_train = df_train[LABEL_COL].tolist()

    X_dev = df_dev[TEXT_COL].tolist()
    y_dev = df_dev[LABEL_COL].tolist()

    X_test = df_test[TEXT_COL].tolist()
    y_test = df_test[LABEL_COL].tolist()

    print("Overlap train-dev :", overlap_count(X_train, X_dev))
    print("Overlap train-test:", overlap_count(X_train, X_test))
    print("Overlap dev-test  :", overlap_count(X_dev, X_test))

    y_train_m, y_dev_m, y_test_m, label2id, id2label = remap_labels(y_train, y_dev, y_test)
    num_labels = len(label2id)
    label_names = [id2label[i] for i in range(num_labels)]
    print("num_labels =", num_labels, "| label2id =", label2id)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label={int(k): v for k, v in id2label.items()},
        label2id={str(k): int(v) for k, v in label2id.items()},
    )

    train_ds = TextDataset(X_train, y_train_m, tokenizer, max_length=HP["max_length"])
    dev_ds   = TextDataset(X_dev,   y_dev_m,   tokenizer, max_length=HP["max_length"])
    test_ds  = TextDataset(X_test,  y_test_m,  tokenizer, max_length=HP["max_length"])

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Early stopping cần save + load best
    save_strategy = "epoch"
    load_best = True if USE_EARLY_STOP else False

    args = make_training_args(
        output_dir=OUTPUT_DIR,
        learning_rate=HP["learning_rate"],
        per_device_train_batch_size=HP["batch_size"],
        per_device_eval_batch_size=HP["batch_size"],
        num_train_epochs=HP["epochs"],
        weight_decay=HP["weight_decay"],
        warmup_ratio=HP["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy=save_strategy,
        save_total_limit=1 if USE_EARLY_STOP else 2,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_f1_macro" if USE_EARLY_STOP else None,
        greater_is_better=True if USE_EARLY_STOP else None,
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        seed=RANDOM_STATE,
        data_seed=RANDOM_STATE,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=HP["grad_accum"],
        # Multi-GPU DDP (khuyến nghị)
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
    )

    callbacks = []
    if USE_EARLY_STOP:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE))

    trainer = make_trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=make_compute_metrics(),
        callbacks=callbacks,
    )

    trainer.train()

    # DEV
    dev_out = trainer.predict(dev_ds)
    dev_pred = np.argmax(dev_out.predictions, axis=-1)
    eval_and_print("DEV", y_dev_m, dev_pred, label_names=label_names)

    # TEST (1 lần)
    test_out = trainer.predict(test_ds)
    test_pred = np.argmax(test_out.predictions, axis=-1)
    eval_and_print("TEST", y_test_m, test_pred, label_names=label_names)

    # SAVE
    Path(SAVE_FINAL_DIR).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(SAVE_FINAL_DIR)
    tokenizer.save_pretrained(SAVE_FINAL_DIR)

    info = {
        "model_name": MODEL_NAME,
        "hyperparams": HP,
        "use_early_stop": bool(USE_EARLY_STOP),
        "early_stop_patience": int(EARLY_STOP_PATIENCE),
        "num_labels": num_labels,
        "label2id": {str(k): int(v) for k, v in label2id.items()},
        "id2label": {str(k): v for k, v in id2label.items()},
        "dev_f1_macro": float(f1_score(y_dev_m, dev_pred, average="macro")),
        "dev_acc": float(accuracy_score(y_dev_m, dev_pred)),
        "test_f1_macro": float(f1_score(y_test_m, test_pred, average="macro")),
        "test_acc": float(accuracy_score(y_test_m, test_pred)),
        "train_path": TRAIN_PATH,
        "dev_path": DEV_PATH,
        "test_path": TEST_PATH,
    }
    with open(SAVE_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"\nSaved final model to: {SAVE_FINAL_DIR}")
    print(f"Saved info to      : {SAVE_INFO_PATH}")

if __name__ == "__main__":
    main()

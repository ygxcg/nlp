"""Task 3: original handout BERT fine-tuning or meeting-note frozen BERT + LR."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from experiment_utils import SEED, metrics, select_and_evaluate, write_json
from task1 import load_data, make_split

ROOT = Path(__file__).resolve().parent


def evaluate(model, loader, device):
    model.eval()
    predictions, truth, loss_sum = [], [], 0.0
    with torch.inference_mode():
        for ids, mask, labels in loader:
            output = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                           labels=labels.to(device))
            loss_sum += output.loss.item() * len(labels)
            predictions.extend(output.logits.argmax(-1).cpu().tolist())
            truth.extend(labels.tolist())
    return np.asarray(predictions), metrics(truth, predictions), loss_sum / len(truth)


def save_predictions(path, indices, truth, predictions, classes, name):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['representation', 'nyt_row_index', 'true_label', 'prediction'])
        writer.writerows((name, int(i), str(y), str(p))
                         for i, y, p in zip(indices, truth, predictions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['finetune', 'frozen'], default='finetune')
    parser.add_argument('--model', help='Local model directory or Hugging Face id')
    parser.add_argument('--data', type=Path, default=ROOT / 'HW-1/nyt.csv')
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'task3_artifacts')
    parser.add_argument('--output', type=Path, default=ROOT / 'task3_results.json')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    model_id = ('google-bert/bert-base-uncased' if args.mode == 'finetune'
                else 'google/bert_uncased_L-2_H-128_A-2')
    source = args.model or model_id
    maximum = 64 if args.mode == 'finetune' else 512
    classes = ['business', 'politics', 'sports']
    texts, labels = load_data(args.data)
    split = make_split(texts, labels)
    split_names = ['train', 'validation', 'test']
    write_json(args.artifacts / 'split_indices.json',
               {name: idx.tolist() for name, idx in zip(split_names, split)})
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=args.local_files_only)
    tokenizer.truncation_side = 'right'
    encoded = tokenizer(texts, max_length=maximum, truncation=True,
                        padding='max_length', return_tensors='pt')
    # Full WordPiece lengths are diagnostic only; no parameters are fit on these texts.
    lengths = np.asarray([len(tokenizer.encode(text, add_special_tokens=True,
                                              truncation=False, verbose=False)) for text in texts])
    targets = torch.tensor([classes.index(str(label)) for label in labels])
    loaders = [DataLoader(TensorDataset(encoded['input_ids'][idx],
                           encoded['attention_mask'][idx], targets[idx]),
                           batch_size=args.batch_size, shuffle=False) for idx in split]
    common = dict(data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  random_seed=SEED, mode=args.mode, model_id=model_id, model_source=source,
                  max_length=maximum, truncation='right; keep first content tokens',
                  split={name: len(idx) for name, idx in zip(split_names, split)},
                  tokenization='pretrained uncased WordPiece; special tokens included',
                  truncated_documents={name: int((lengths[idx] > maximum).sum())
                                       for name, idx in zip(split_names, split)},
                  device=device, threads=args.threads, batch_size=args.batch_size,
                  environment={'torch': torch.__version__,
                               'transformers': __import__('transformers').__version__,
                               'numpy': np.__version__,
                               'scikit-learn': __import__('sklearn').__version__})
    if Path(source).is_dir():
        common['pretrained_file_sha256'] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(source).iterdir() if path.is_file()}
    started = time.perf_counter()
    if args.mode == 'frozen':
        model = AutoModel.from_pretrained(source, local_files_only=args.local_files_only).to(device)
        model.requires_grad_(False).eval()
        matrices = []
        for name, loader in zip(split_names, loaders):
            vectors = []
            with torch.inference_mode():
                for step, (ids, mask, _) in enumerate(loader):
                    hidden = model(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state
                    content = mask.to(device).bool() & (ids.to(device) != tokenizer.cls_token_id) & (ids.to(device) != tokenizer.sep_token_id)
                    pooled = (hidden * content.unsqueeze(-1)).sum(1) / content.sum(1).clamp_min(1).unsqueeze(-1)
                    vectors.append(pooled.cpu().numpy())
                    if step % 25 == 0:
                        print(f'Encoding {name}: {step + 1}/{len(loader)} batches', flush=True)
            matrix = np.concatenate(vectors)
            np.save(args.artifacts / f'{name}_features.npy', matrix)
            matrices.append(matrix)
        result, _, valid_pred, test_pred = select_and_evaluate(
            'Frozen BERT-Tiny + Logistic Regression', matrices,
            [labels[idx] for idx in split],
            {'encoder_frozen': True, 'pooling': 'last layer mean excluding PAD/CLS/SEP',
             'feature_dimension': model.config.hidden_size}, args.artifacts, 'bert')
        common['parameter_count'] = sum(p.numel() for p in model.parameters())
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            source, num_labels=3, id2label=dict(enumerate(classes)),
            label2id={label: i for i, label in enumerate(classes)},
            local_files_only=args.local_files_only).to(device)
        common['parameter_count'] = sum(p.numel() for p in model.parameters())
        no_decay = ('bias', 'LayerNorm.weight')
        optimizer = torch.optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if not any(t in n for t in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(t in n for t in no_decay)], 'weight_decay': 0.0},
        ], lr=2e-5)
        train_loader = DataLoader(loaders[0].dataset, batch_size=args.batch_size,
                                  shuffle=True, generator=torch.Generator().manual_seed(SEED))
        history, best_key, best_epoch = [], None, None
        checkpoint = args.artifacts / 'best_model'
        for epoch in range(1, 4):
            model.train()
            epoch_start, loss_sum = time.perf_counter(), 0.0
            for step, (ids, mask, y) in enumerate(train_loader):
                optimizer.zero_grad(set_to_none=True)
                loss = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=y.to(device)).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += loss.item() * len(y)
                if step % 20 == 0:
                    print(f'Epoch {epoch}/3 batch {step + 1}/{len(train_loader)} loss={loss.item():.4f} elapsed={time.perf_counter()-epoch_start:.1f}s', flush=True)
            _, score, valid_loss = evaluate(model, loaders[1], device)
            row = {'epoch': epoch, 'train_loss': loss_sum / len(split[0]),
                   'validation_loss': valid_loss, 'validation_accuracy': score['accuracy'],
                   'validation_macro_f1': score['macro_f1'],
                   'seconds': time.perf_counter() - epoch_start}
            history.append(row)
            write_json(args.artifacts / 'training_history.json', history)
            print(json.dumps(row), flush=True)
            key = (score['macro_f1'], score['accuracy'], -epoch)
            if best_key is None or key > best_key:
                best_key, best_epoch = key, epoch
                model.save_pretrained(checkpoint)
                tokenizer.save_pretrained(checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
        valid_ids, valid_score, _ = evaluate(model, loaders[1], device)
        test_ids, test_score, _ = evaluate(model, loaders[2], device)
        valid_pred, test_pred = np.asarray(classes)[valid_ids], np.asarray(classes)[test_ids]
        result = {'representation': 'BERT-base-uncased fine-tuning', 'encoder_frozen': False,
                  'pooling': 'last-layer CLS -> BERT pooler -> dropout -> linear classification head',
                  'epochs': 3, 'selected_epoch': best_epoch, 'optimizer': 'AdamW',
                  'learning_rate': 2e-5, 'weight_decay': 0.01, 'gradient_clip_norm': 1.0,
                  'scheduler': None, 'class_weight': None,
                  'selection_rule': 'validation macro-F1, then accuracy, then earlier epoch',
                  'training_history': history, 'validation_accuracy': valid_score['accuracy'],
                  'validation_macro_f1': valid_score['macro_f1'],
                  'test_accuracy': test_score['accuracy'], 'test_macro_f1': test_score['macro_f1'],
                  'classes': classes,
                  'test_confusion_matrix': confusion_matrix(labels[split[2]], test_pred, labels=classes).tolist(),
                  'test_classification_report': classification_report(labels[split[2]], test_pred, labels=classes, output_dict=True, zero_division=0)}
    for name, idx, pred in [('validation', split[1], valid_pred), ('test', split[2], test_pred)]:
        save_predictions(args.artifacts / f'{name}_predictions.csv', idx, labels[idx], pred, classes, result['representation'])
    bad = [{'nyt_row_index': int(i), 'true_label': str(labels[i]), 'prediction': str(pred),
            'text': texts[i], 'visible_text': tokenizer.decode(encoded['input_ids'][i], skip_special_tokens=True),
            'full_wordpiece_length': int(lengths[i])}
           for i, pred in zip(split[2], test_pred) if labels[i] != pred]
    write_json(args.artifacts / 'bad_cases.json', bad)
    common['elapsed_seconds'] = time.perf_counter() - started
    common['experiments'] = [result]
    write_json(args.output, common)
    print(f'FINAL accuracy={result["test_accuracy"]:.4f} macro-F1={result["test_macro_f1"]:.4f}', flush=True)


if __name__ == '__main__':
    main()

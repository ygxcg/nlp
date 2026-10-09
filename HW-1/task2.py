"""Task 2：使用词向量完成 NYT 新闻分类。"""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
import platform
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Mapping



import numpy as np
import gensim
import sklearn
import scipy
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from threadpoolctl import threadpool_limits
from sklearn.linear_model import LogisticRegression
from experiment_utils import CLASSIFIER_SETTINGS, C_CANDIDATES, select_and_evaluate


SEED = 42
DIMENSION = 100
TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")
ROOT = Path(__file__).resolve().parent
NYT_PATH = ROOT / "HW-1" / "nyt.csv"
AG_PATH = ROOT / "HW-1" / "ag.csv"
GLOVE_FILE = ROOT / "glove.6B.100d.txt"
GLOVE_URL = "https://huggingface.co/kcz358/glove/resolve/main/glove.6B/glove.6B.100d.txt"
DEFAULT_OUTPUT = ROOT / "task2_results.json"


def tokenize(text: str) -> list[str]:
    """将文本转为小写 token 序列。"""
    return TOKEN_PATTERN.findall(text.lower())


def load_nyt(path: Path) -> tuple[list[str], np.ndarray]:
    """读取 NYT 文本和标签。"""
    texts: list[str] = []
    labels: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if text and label:
                texts.append(text)
                labels.append(label)
    return texts, np.asarray(labels)


def load_ag(path: Path) -> list[str]:
    """读取 AG News 文本；该数据集只用于训练词向量。"""
    texts: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            text = (row.get("text") or "").strip()
            if text:
                texts.append(text)
    return texts


def split_indices(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """复用 Task1 的固定分层 80/10/10 划分。"""
    from task1 import make_split
    return make_split(np.arange(len(labels)), labels)


def ensure_glove() -> Path:
    """下载到临时文件，验证完整性后替换，保留原有残缺文件。"""
    if GLOVE_FILE.exists():
        try:
            load_glove(GLOVE_FILE, set())
            return GLOVE_FILE
        except ValueError as error:
            print(error, flush=True)
    temporary = GLOVE_FILE.with_suffix('.txt.download')
    print("开始下载完整 GloVe 100 维向量...", flush=True)
    request = urllib.request.Request(GLOVE_URL, headers={'User-Agent': 'NLP-homework/1.0'})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as handle:
        downloaded = 0
        last_message = time.perf_counter()
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
            downloaded += len(chunk)
            if time.perf_counter() - last_message >= 10:
                print(f'GloVe: {downloaded / 1024**2:.1f} MiB', flush=True)
                last_message = time.perf_counter()
    load_glove(temporary, set())
    if GLOVE_FILE.exists():
        backup = GLOVE_FILE.with_suffix('.txt.incomplete')
        if backup.exists():
            raise FileExistsError(f'备份已存在：{backup}；未替换原文件')
        GLOVE_FILE.rename(backup)
    temporary.replace(GLOVE_FILE)
    return GLOVE_FILE


def load_glove(path: Path, needed_words: set[str]) -> dict[str, np.ndarray]:
    """只加载当前语料需要的 GloVe 词向量，降低内存占用。"""
    vectors: dict[str, np.ndarray] = {}
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for rows, line in enumerate(handle, start=1):
            fields = line.split()
            if len(fields) != DIMENSION + 1:
                raise ValueError(f'GloVe 第 {rows} 行不完整或不是 100 维：{path}')
            if fields[0] not in needed_words:
                continue
            vector = np.asarray(fields[1:], dtype=np.float32)
            if not np.isfinite(vector).all():
                raise ValueError(f'GloVe 第 {rows} 行包含非有限数值')
            vectors[fields[0]] = vector
    if rows != 400000:
        raise ValueError(f'GloVe 文件不完整：{rows:,} 行，应为 400,000 行')
    return vectors


class EpochProgress(CallbackAny2Vec):
    def __init__(self, name):
        self.name = name
        self.epoch = 0
        self.started = time.perf_counter()
        self.previous_loss = 0.0
        self.history = []

    def on_epoch_begin(self, model):
        # Reset reporting only, not weights/optimizer: avoid float32 cumulative
        # loss saturation across a multi-epoch run on these large corpora.
        model.running_training_loss = 0.0

    def on_epoch_end(self, model):
        self.epoch += 1
        loss = float(model.get_latest_training_loss())
        self.previous_loss += loss
        cumulative = self.previous_loss
        self.history.append({'epoch': self.epoch, 'epoch_loss': loss, 'cumulative_loss': cumulative})
        print(f'{self.name}: epoch {self.epoch}/{model.epochs}, '
              f'loss={loss:.1f}, elapsed={time.perf_counter() - self.started:.1f}s', flush=True)


def stable_hash(word):
    """使初始化不依赖 Python 进程随机化的 hash。"""
    return int.from_bytes(hashlib.md5(word.encode('utf-8')).digest()[:4], 'little')


def train_word2vec(documents, epochs=5, name='Word2Vec', artifacts=None):
    """gensim Skip-gram + 负采样；完整语料训练，不截断词对。"""
    progress = EpochProgress(name)
    model = Word2Vec(
        sentences=documents, vector_size=DIMENSION, sg=1, window=5,
        min_count=3, negative=5, hs=0, sample=1e-3, epochs=epochs,
        workers=1, seed=SEED, hashfxn=stable_hash, sorted_vocab=1,
        alpha=0.025, min_alpha=0.0001, shrink_windows=True,
        compute_loss=True, ns_exponent=0.75, callbacks=[progress],
    )
    if artifacts is not None:
        model.wv.save(str(artifacts / f'{name.lower()}_word2vec.kv'))
        (artifacts / f'{name.lower()}_word2vec_loss.json').write_text(
            json.dumps(progress.history, indent=2), encoding='utf-8')
    return model.wv


def document_vectors(token_documents: list[list[str]], vectors: Mapping[str, np.ndarray] | KeyedVectors) -> np.ndarray:
    """对文档中所有已登录词向量取平均。"""
    output = np.zeros((len(token_documents), DIMENSION), dtype=np.float32)
    for index, document in enumerate(token_documents):
        valid = [vectors[word] for word in document if word in vectors]
        if valid:
            output[index] = np.mean(valid, axis=0)
    return output


def file_sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def coverage_statistics(documents, vectors):
    total = sum(map(len, documents))
    covered = sum(word in vectors for document in documents for word in document)
    words = {word for document in documents for word in document}
    return {
        'total_tokens': total, 'covered_tokens': covered,
        'token_coverage': covered / total if total else 0.0,
        'unique_words': len(words), 'covered_words': sum(word in vectors for word in words),
        'empty_documents': sum(not any(word in vectors for word in doc) for doc in documents),
    }


def evaluate(name, train_x, valid_x, test_x, train_y, valid_y, test_y, details, artifacts):
    """Select C using validation only, plot LR learning curves, then evaluate test once."""
    stem = "glove" if name == "Pre-trained GloVe" else "ag" if "AG News" in name else "nyt"
    result, _, valid_pred, test_pred = select_and_evaluate(
        name, (train_x, valid_x, test_x), (train_y, valid_y, test_y), details, artifacts, stem)
    result["test_predictions"] = test_pred.tolist()
    result["validation_predictions"] = valid_pred.tolist()
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nyt", type=Path, default=NYT_PATH)
    parser.add_argument("--ag", type=Path, default=AG_PATH)
    parser.add_argument("--glove", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'task2_artifacts')
    parser.add_argument('--prepare-glove', action='store_true', help='验证/下载 GloVe 后退出')
    args = parser.parse_args()
    if args.prepare_glove:
        print(ensure_glove(), flush=True)
        return
    if args.epochs < 1:
        parser.error('--epochs 必须大于 0')
    np.random.seed(SEED)
    # 100 维小矩阵使用单线程 BLAS，减少大量线程的调度开销。
    thread_limiter = threadpool_limits(limits=1)
    args.artifacts.mkdir(parents=True, exist_ok=True)

    texts, labels = load_nyt(args.nyt)
    ag_texts = load_ag(args.ag)
    train_idx, valid_idx, test_idx = split_indices(labels)
    token_documents = [tokenize(text) for text in texts]
    train_tokens = [token_documents[index] for index in train_idx]
    valid_tokens = [token_documents[index] for index in valid_idx]
    test_tokens = [token_documents[index] for index in test_idx]
    train_y, valid_y, test_y = labels[train_idx], labels[valid_idx], labels[test_idx]
    results = []
    print(f'NYT={len(texts)}, AG={len(ag_texts)}; split='
          f'{len(train_idx)}/{len(valid_idx)}/{len(test_idx)}', flush=True)
    (args.artifacts / 'split_indices.json').write_text(json.dumps({
        name: subset.tolist() for name, subset in zip(
            ('train', 'validation', 'test'), (train_idx, valid_idx, test_idx))
    }), encoding='utf-8')

    def coverage(vectors):
        return {name: coverage_statistics(documents, vectors) for name, documents in zip(
            ('train', 'validation', 'test'), (train_tokens, valid_tokens, test_tokens))}

    all_words = set(word for document in token_documents for word in document)
    glove_path = args.glove or ensure_glove()
    glove_vectors = load_glove(glove_path, all_words)
    print('GloVe verified: 400,000 rows', flush=True)
    glove_train = document_vectors(train_tokens, glove_vectors)
    glove_valid = document_vectors(valid_tokens, glove_vectors)
    glove_test = document_vectors(test_tokens, glove_vectors)
    results.append(evaluate(
        "Pre-trained GloVe",
        glove_train, glove_valid, glove_test, train_y, valid_y, test_y,
        {"embedding_dimension": DIMENSION, "covered_words": len(glove_vectors),
         'glove_bytes': glove_path.stat().st_size,
         'glove_sha256': file_sha256(glove_path),
         'coverage': coverage(glove_vectors)}, args.artifacts,
    ))

    ag_tokens = [tokenize(text) for text in ag_texts]
    start = time.perf_counter()
    ag_vectors = train_word2vec(ag_tokens, args.epochs, 'AG', args.artifacts)
    ag_seconds = time.perf_counter() - start
    ag_train = document_vectors(train_tokens, ag_vectors)
    ag_valid = document_vectors(valid_tokens, ag_vectors)
    ag_test = document_vectors(test_tokens, ag_vectors)
    results.append(evaluate(
        "Word2Vec trained on AG News",
        ag_train, ag_valid, ag_test, train_y, valid_y, test_y,
        {"embedding_dimension": DIMENSION, "embedding_vocabulary": len(ag_vectors),
         'embedding_train_documents': len(ag_tokens),
         'embedding_train_tokens': sum(map(len, ag_tokens)),
         'embedding_train_seconds': round(ag_seconds, 3), 'coverage': coverage(ag_vectors)}, args.artifacts,
    ))

    start = time.perf_counter()
    nyt_vectors = train_word2vec(train_tokens, args.epochs, 'NYT', args.artifacts)
    nyt_seconds = time.perf_counter() - start
    nyt_train = document_vectors(train_tokens, nyt_vectors)
    nyt_valid = document_vectors(valid_tokens, nyt_vectors)
    nyt_test = document_vectors(test_tokens, nyt_vectors)
    results.append(evaluate(
        "Word2Vec trained on NYT",
        nyt_train, nyt_valid, nyt_test, train_y, valid_y, test_y,
        {"embedding_dimension": DIMENSION, "embedding_vocabulary": len(nyt_vectors),
         'embedding_train_documents': len(train_tokens),
         'embedding_train_tokens': sum(map(len, train_tokens)),
         'embedding_train_seconds': round(nyt_seconds, 3), 'coverage': coverage(nyt_vectors)}, args.artifacts,
    ))

    output = {
        "random_seed": SEED,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__,
                        'gensim': gensim.__version__},
        'label_distribution': dict(Counter(labels.tolist())),
        'nyt_embedding_training_scope': 'NYT training split only',
        'preprocessing': {'lowercase': True, 'tokenization': TOKEN_PATTERN.pattern,
                          'remove_punctuation': True, 'remove_stopwords': False,
                          'remove_numbers': False, 'min_token_length': 2, 'truncate': False},
        'tokenization': TOKEN_PATTERN.pattern,
        'data': {'nyt': str(args.nyt), 'ag': str(args.ag),
                 'nyt_sha256': file_sha256(args.nyt), 'ag_sha256': file_sha256(args.ag)},
        "split": {"train": len(train_idx), "validation": len(valid_idx), "test": len(test_idx)},
        "ag_documents": len(ag_texts),
        "word2vec": {'algorithm': 'Skip-gram with negative sampling', 'vector_size': DIMENSION,
                     'window': 5, 'min_count': 3, 'negative_samples': 5, 'epochs': args.epochs,
                     'workers': 1, 'seed': SEED, 'sample': 1e-3, 'alpha': 0.025,
                     'min_alpha': 0.0001, 'shrink_windows': True, 'compute_loss': True,
                     'loss_reporting': 'reset loss accumulator at each epoch; parameters unchanged',
                     'ns_exponent': 0.75},
        "classifier": CLASSIFIER_SETTINGS,
        "C_candidates": list(C_CANDIDATES),
        "experiments": results,
    }
    with (args.artifacts / 'test_predictions.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['representation', 'nyt_row_index', 'true_label', 'predicted_label'])
        for result in results:
            predictions = result.pop('test_predictions')
            writer.writerows(zip([result['representation']] * len(test_idx),
                                 test_idx.tolist(), test_y.tolist(), predictions))
    with (args.artifacts / 'validation_predictions.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['representation', 'nyt_row_index', 'true_label', 'predicted_label'])
        for result in results:
            predictions = result.pop('validation_predictions')
            writer.writerows(zip([result['representation']] * len(valid_idx),
                                 valid_idx.tolist(), valid_y.tolist(), predictions))
    for name in ('ag', 'nyt'):
        history_path = args.artifacts / f'{name}_word2vec_loss.json'
        output[f'{name}_word2vec_loss'] = json.loads(history_path.read_text(encoding='utf-8'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    thread_limiter.restore_original_limits()
    print(f"结果已保存到 {args.output}")


if __name__ == "__main__":
    main()

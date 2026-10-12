"""Pinned MiniLM ONNX inference, mean pooling, and unit-length embeddings."""
import hashlib
import json

import numpy as np

from backend.config import Settings
from backend.errors import DocumentError

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingEngine:
    dimension = 384
    max_sequence_length = 256
    model_id = MODEL_ID

    def __init__(self, settings: Settings):
        self.settings = settings
        self.tokenizer = None
        self.session = None

    def load(self):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            folder = self.settings.model_dir
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            if manifest["model"] != MODEL_ID or manifest["revision"] != self.settings.embedding_revision:
                raise ValueError("Embedding model revision mismatch")
            for filename, expected in manifest["checksums"].items():
                if filename not in {"model.onnx", "tokenizer.json", "config.json", "pooling_config.json"}:
                    raise ValueError("Unexpected model manifest path")
                with (folder / filename).open("rb") as file:
                    if hashlib.file_digest(file, "sha256").hexdigest() != expected:
                        raise ValueError("Embedding model checksum mismatch")
            pooling = json.loads((folder / "pooling_config.json").read_text(encoding="utf-8"))
            if not pooling.get("pooling_mode_mean_tokens") or pooling.get("pooling_mode_cls_token"):
                raise ValueError("Unsupported pooling configuration")
            self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
            self.tokenizer.no_truncation()
            self.tokenizer.no_padding()
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.settings.embedding_threads
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(folder / "model.onnx"),
                sess_options=options, providers=["CPUExecutionProvider"])
            configuration = {"model": MODEL_ID, "revision": manifest["revision"],
                "tokenizer": manifest["checksums"]["tokenizer.json"], "dimension": self.dimension,
                "chunk_tokens": self.settings.chunk_tokens, "overlap": self.settings.chunk_overlap,
                "chunker_version": 1, "pooling": "masked_mean_l2"}
            self.fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
        except (ImportError, OSError, ValueError, KeyError, RuntimeError) as error:
            self.session = None
            raise DocumentError("embedding_unavailable", "Local embeddings are not ready. "
                "Run python scripts/setup_models.py and restart the backend.", 503) from error

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.session is None:
            raise DocumentError("embedding_unavailable", "Local embeddings are not loaded.", 503)
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = []
        for start in range(0, len(texts), 32):
            encodings = self.tokenizer.encode_batch(texts[start:start + 32])
            if any(len(item.ids) > self.max_sequence_length for item in encodings):
                raise DocumentError("question_too_long", "The question exceeds the embedding model's "
                    "256-token limit. Ask a shorter question.")
            width = max(len(item.ids) for item in encodings)
            ids = np.zeros((len(encodings), width), dtype=np.int64)
            mask = np.zeros_like(ids)
            types = np.zeros_like(ids)
            for row, item in enumerate(encodings):
                ids[row, :len(item.ids)] = item.ids
                mask[row, :len(item.ids)] = item.attention_mask
                types[row, :len(item.ids)] = item.type_ids
            inputs = {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
            outputs = self.session.run(None, {entry.name: inputs[entry.name]
                                             for entry in self.session.get_inputs()})
            hidden = outputs[0]
            if hidden.ndim != 3 or hidden.shape[-1] != self.dimension:
                raise DocumentError("embedding_failed", "The embedding model returned an invalid shape.", 503)
            weights = mask[:, :, None].astype(np.float32)
            pooled = (hidden * weights).sum(axis=1) / weights.sum(axis=1).clip(min=1)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            if not np.isfinite(pooled).all() or (norms <= 0).any():
                raise DocumentError("embedding_failed", "The embedding model returned invalid vectors.", 503)
            vectors.append((pooled / norms).astype(np.float32))
        return np.ascontiguousarray(np.concatenate(vectors))

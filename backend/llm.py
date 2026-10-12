"""Local generator adapters. Provider failures are never treated as missing evidence."""
import json
import time

import httpx

from backend.errors import DocumentError
from backend.schemas import GeneratedAnswer


class LocalGenerator:
    def __init__(self, settings):
        self.settings = settings
        self.model = settings.llm_model
        self.url = settings.llm_url.rstrip("/")
        self._status = None
        self._checked_at = 0

    def request(self, path, payload=None, timeout=None):
        try:
            with httpx.Client(timeout=httpx.Timeout(timeout or self.settings.llm_timeout_seconds,
                                                  connect=5), trust_env=False) as client:
                if payload is None:
                    response = client.get(self.url + path)
                else:
                    response = client.post(self.url + path, json=payload)
                if response.is_error:
                    raise DocumentError("generator_error", "The local answer model returned an error. "
                        "Check the model server log and retry.", 502)
                if len(response.content) > 1024 * 1024:
                    raise ValueError("Oversized provider response")
                result = response.json()
                if not isinstance(result, dict):
                    raise ValueError("Expected a provider response object")
                return result
        except httpx.TimeoutException as error:
            raise DocumentError("generator_timeout", "The local model took too long to answer. "
                "Try a shorter question or fewer source passages.", 504) from error
        except httpx.HTTPError as error:
            raise DocumentError("generator_unavailable", "Start the local model with "
                "scripts/run-model.ps1, then try again. Search remains available.", 503) from error
        except ValueError as error:
            raise DocumentError("generator_invalid_response", "The model server returned an invalid response.", 502) from error

    def status(self):
        if self._status is not None and time.monotonic() - self._checked_at < 10:
            return self._status
        ready = False
        try:
            if self.settings.llm_provider == "llamacpp":
                data = self.request("/v1/models", timeout=3)
                ready = self.model in {entry["id"] for entry in data.get("data", [])}
            else:
                data = self.request("/api/tags", timeout=3)
                ready = self.model in {entry["name"] for entry in data.get("models", [])}
        except (DocumentError, KeyError, TypeError):
            pass
        self._status = {"state": "ready" if ready else "unavailable",
            "provider": self.settings.llm_provider, "model": self.model}
        self._checked_at = time.monotonic()
        return self._status

    def count_prompt_tokens(self, messages):
        if self.settings.llm_provider == "llamacpp":
            try:
                prompt = self.request("/apply-template", {"messages": messages}, timeout=10)["prompt"]
                if not isinstance(prompt, str):
                    raise ValueError("Expected a formatted prompt")
                tokens = self.request("/tokenize", {"content": prompt, "add_special": True,
                                                   "parse_special": True}, timeout=10)["tokens"]
                if not isinstance(tokens, list) or any(not isinstance(token, int) for token in tokens):
                    raise ValueError("Expected model token IDs")
                return len(tokens)
            except (KeyError, TypeError, ValueError) as error:
                raise DocumentError("generator_invalid_response", "The model tokenizer returned an invalid response.", 502) from error
        # Ollama has no tokenizer endpoint. UTF-8 byte count plus template overhead is
        # a conservative budget for supported byte-level instruction-model tokenizers.
        return sum(len(message["content"].encode("utf-8")) for message in messages) + 512

    def generate(self, messages):
        schema = GeneratedAnswer.model_json_schema()
        source_ids = [source["source_id"] for source in json.loads(messages[1]["content"])["sources"]]
        schema["properties"]["citations"]["items"] = {"type": "string", "enum": source_ids}
        if self.settings.llm_provider == "llamacpp":
            prompt = self.request("/apply-template", {"messages": messages}, timeout=10)["prompt"]
            result = self.request("/completion", {
                "prompt": prompt, "stream": False, "temperature": 0, "seed": 7,
                "n_predict": self.settings.llm_output_tokens, "json_schema": schema,
                "cache_prompt": True,
            })
            try:
                if result.get("stop_type") == "limit" or result.get("truncated"):
                    raise DocumentError("generation_incomplete", "The answer exceeded its output budget. "
                        "Ask a more focused question.", 502)
                return result["content"]
            except (KeyError, IndexError, TypeError) as error:
                raise DocumentError("generator_invalid_response", "The model server returned an invalid response.", 502) from error
        result = self.request("/api/chat", {
            "model": self.model, "messages": messages, "stream": False, "format": schema,
            "options": {"temperature": 0, "seed": 7, "num_ctx": self.settings.llm_context_tokens,
                        "num_predict": self.settings.llm_output_tokens},
        })
        try:
            if result.get("done_reason") == "length":
                raise DocumentError("generation_incomplete", "The answer exceeded its output budget.", 502)
            return result["message"]["content"]
        except (KeyError, TypeError) as error:
            raise DocumentError("generator_invalid_response", "The model server returned an invalid response.", 502) from error

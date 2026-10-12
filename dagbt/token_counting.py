"""Opt-in, fail-closed token counts for the verified deployment chat path.

Only final, adapted BridgeTree text-chat wires using the server's default chat
template are supported. This module never changes a budget or generates text.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
import errno
import hashlib
import http.client
import json
import math
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

from .model_runtime import is_bridgetree, resolve_api_key


ESTIMATOR_ID = "deployment_vllm_chat_v1"
MAX_CACHE_ENTRIES = 512
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_IDENTICAL_ATTEMPTS = 10
RETRYABLE_HTTP_STATUSES = frozenset({500, 502, 503, 504})
RETRYABLE_ERRNOS = frozenset({errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNREFUSED,
    errno.ECONNABORTED, errno.EPIPE, errno.ENETRESET, errno.ENETUNREACH, errno.EHOSTUNREACH})
# These affect generation only, not the input chat template. Keep this list
# deliberately small: unfamiliar provider options require separate validation.
SAMPLING_FIELDS = frozenset({"temperature", "max_tokens", "top_p", "seed",
                             "presence_penalty", "frequency_penalty",
                             "repetition_penalty", "top_k", "min_p", "stop"})
WIRE_FIELDS = SAMPLING_FIELDS | {"model", "messages"}


class TokenizationError(ValueError):
    """Safe failure without provider bodies, request text, or credentials."""

    def __init__(self, category, *, http_status=None):
        self.category = category
        self.http_status = http_status
        super().__init__("Deployment tokenization failed: " + category)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _origin(url):
    try:
        parsed = urllib.parse.urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment):
            raise ValueError
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except (TypeError, ValueError):
        raise TokenizationError("invalid_endpoint") from None
    return parsed, (parsed.scheme, parsed.hostname.lower(), port)


def validate_tokenization_config(config):
    """Validate explicit opt-in without network access; absent means disabled."""
    if "tokenization" not in config:
        return None
    options = config["tokenization"]
    if (not isinstance(options, Mapping)
            or set(options) - {"mode", "endpoint", "timeout_seconds"}
            or options.get("mode") != "vllm_chat"):
        raise TokenizationError("invalid_config")
    if not is_bridgetree(config):
        raise TokenizationError("unsupported_model_profile")
    if not isinstance(config.get("llm_model"), str) or not config["llm_model"].strip():
        raise TokenizationError("invalid_model")
    endpoint, token_origin = _origin(options.get("endpoint"))
    _, chat_origin = _origin(config.get("llm_base_url"))
    if token_origin != chat_origin or endpoint.path != "/tokenize":
        raise TokenizationError("endpoint_must_be_same_origin_root_tokenize")
    timeout = options.get("timeout_seconds", 15)
    if (isinstance(timeout, bool) or not isinstance(timeout, (float, int))
            or not math.isfinite(timeout) or timeout <= 0):
        raise TokenizationError("invalid_timeout")
    attempts = config.get("max_identical_attempts", 3)
    if type(attempts) is not int or not 1 <= attempts <= MAX_IDENTICAL_ATTEMPTS:
        raise TokenizationError("invalid_max_identical_attempts")
    fusion = config.get("fusion", {})
    if not isinstance(fusion, Mapping) or fusion.get("response_format", "plain") != "plain":
        raise TokenizationError("unsupported_response_protocol")
    context = fusion.get("context_tokens", 16384)
    if type(context) is not int or context <= 0:
        raise TokenizationError("invalid_context_limit")
    provider = config.get("llm_provider_request_params", {})
    if not isinstance(provider, Mapping) or set(provider) - SAMPLING_FIELDS:
        raise TokenizationError("unsupported_provider_parameters")
    try:
        _canonical(provider)
    except (TypeError, ValueError, OverflowError):
        raise TokenizationError("invalid_provider_parameters") from None
    return {"mode": "vllm_chat", "endpoint": endpoint.geturl(),
            "timeout_seconds": float(timeout), "max_identical_attempts": attempts}


def _transient_connection_error(error):
    """Only transport failures known to be transient; no text matching."""
    if isinstance(error, urllib.error.URLError):
        error = error.reason
    if isinstance(error, ssl.SSLError):
        return False
    if isinstance(error, socket.gaierror):
        return error.errno == socket.EAI_AGAIN
    return (isinstance(error, (TimeoutError, ConnectionError, http.client.IncompleteRead))
            or isinstance(error, OSError) and error.errno in RETRYABLE_ERRNOS)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Even a same-origin redirect could alter the documented API semantics.
        raise TokenizationError("redirect_not_allowed", http_status=code)


class DeploymentTokenCounter:
    """One counter per transport/run; its bounded cache never persists counts.

    ``record_event`` receives exactly one safe tokenization_count event per
    count() invocation, including errors and cache hits. The caller owns any
    persistence and ledger integration; this class never reserves LLM/ANN cost.
    ``output`` is accepted for the construction interface but writes no files.
    """

    def __init__(self, config, record_event, output=None):
        self.options = validate_tokenization_config(config)
        if self.options is None:
            raise TokenizationError("not_enabled")
        self.config = deepcopy(dict(config))
        self.model = config["llm_model"]
        self.context_limit = config.get("fusion", {}).get("context_tokens", 16384)
        self.record_event = record_event
        self._cache = OrderedDict()
        # Tokenization must not leak the endpoint-bound bearer key through an
        # ambient HTTP proxy or a redirect to another origin.
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect())

    def _payload(self, wire):
        if not isinstance(wire, Mapping) or set(wire) - WIRE_FIELDS:
            raise TokenizationError("unsupported_wire_fields")
        if wire.get("model") != self.model:
            raise TokenizationError("model_mismatch")
        messages = wire.get("messages")
        if not isinstance(messages, list) or not messages:
            raise TokenizationError("invalid_messages")
        for message in messages:
            if (not isinstance(message, Mapping) or set(message) != {"role", "content"}
                    or message["role"] not in {"system", "user", "assistant"}
                    or not isinstance(message["content"], str)):
                raise TokenizationError("unsupported_message_shape")
        try:
            _canonical(wire)
        except (TypeError, ValueError, OverflowError):
            raise TokenizationError("invalid_wire_json") from None
        # Plain-mode schemas already reside in the final system message. Never
        # tokenize the transport JSON or a guessed local rendering instead.
        return {"model": self.model, "messages": deepcopy(messages),
                "add_generation_prompt": True}

    def _fetch(self, request, event):
        """Retry the same read-only count wire, never a generation request."""
        limit = self.options["max_identical_attempts"]
        for index in range(1, limit + 1):
            started = time.monotonic()
            attempt = {"attempt": index, "status": "error", "retryable": False,
                       "will_retry": False}
            event["attempts"].append(attempt)
            event["http_attempts"] = index
            event["http_retries"] = index - 1
            # The last attempt's status must not inherit a previous HTTP status.
            event.pop("http_status", None)
            failure = None
            try:
                with self._opener.open(request, timeout=self.options["timeout_seconds"]) as response:
                    status = response.getcode()
                    attempt["http_status"] = event["http_status"] = status
                    if status != 200:
                        raise TokenizationError("unexpected_http_status", http_status=status)
                    raw = response.read(MAX_RESPONSE_BYTES + 1)
                event["successful_http_requests"] += 1
                attempt["status"] = "ok"
                return raw
            except urllib.error.HTTPError as exc:
                attempt["http_status"] = event["http_status"] = exc.code
                attempt["retryable"] = exc.code in RETRYABLE_HTTP_STATUSES
                failure = TokenizationError("http_error", http_status=exc.code)
                exc.close()
            except TokenizationError as exc:
                failure = exc
                if exc.http_status is not None:
                    attempt["http_status"] = event["http_status"] = exc.http_status
                attempt["retryable"] = (exc.category == "unexpected_http_status"
                    and exc.http_status in RETRYABLE_HTTP_STATUSES)
            except (urllib.error.URLError, OSError, http.client.IncompleteRead) as exc:
                failure = TokenizationError("connection_or_timeout")
                attempt["retryable"] = _transient_connection_error(exc)
            except Exception:
                failure = TokenizationError("unexpected_transport_error")
            finally:
                attempt["seconds"] = max(0.0, time.monotonic() - started)
            event["failed_http_attempts"] += 1
            attempt["error_category"] = failure.category
            attempt["will_retry"] = attempt["retryable"] and index < limit
            if not attempt["will_retry"]:
                raise failure from None
            delay = min(0.25 * index, 1.0)
            attempt["retry_delay_seconds"] = delay
            time.sleep(delay)

    def count(self, wire):
        started = time.monotonic()
        event = {"event": "tokenization_count", "kind": "tokenization_http",
                 "cache_hit": False, "http_attempts": 0,
                 "http_retries": 0, "failed_http_attempts": 0, "attempts": [],
                 "successful_http_requests": 0, "status": "error",
                 "token_estimator_id": ESTIMATOR_ID}
        try:
            payload = self._payload(wire)
            identity = {"endpoint": self.options["endpoint"],
                        "mode": self.options["mode"], "payload": payload}
            key = _digest(identity)
            event["tokenization_request_sha256"] = key
            wire_hash = _digest(wire)
            if key in self._cache:
                result = deepcopy(self._cache[key])
                self._cache.move_to_end(key)
                event["cache_hit"] = True
            else:
                headers = {"Content-Type": "application/json"}
                try:
                    secret = resolve_api_key(self.config, "llm")
                except Exception:
                    raise TokenizationError("credential_resolution_failed") from None
                if secret:
                    headers["Authorization"] = "Bearer " + secret
                request = urllib.request.Request(self.options["endpoint"],
                    data=_canonical(payload).encode("utf-8"), headers=headers, method="POST")
                raw = self._fetch(request, event)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise TokenizationError("response_too_large")
                try:
                    body = json.loads(raw)
                except (TypeError, ValueError, UnicodeError):
                    raise TokenizationError("invalid_response_json") from None
                if not isinstance(body, Mapping):
                    raise TokenizationError("invalid_response_shape")
                count, maximum = body.get("count"), body.get("max_model_len")
                if type(count) is not int or count < 0 or type(maximum) is not int or maximum <= 0:
                    raise TokenizationError("invalid_response_counts")
                event.update(count=count, max_model_len=maximum)
                if maximum < self.context_limit:
                    raise TokenizationError("deployment_context_below_configured_limit")
                result = {"count": count, "max_model_len": maximum,
                    "tokenization_request_sha256": key,
                    "tokenization_endpoint_sha256": _digest(self.options["endpoint"]),
                    "token_count_is_estimate": False, "token_estimator_id": ESTIMATOR_ID,
                    "tokenizer_id": None, "add_generation_prompt": True,
                    "tokenization_scope": "adapted_plain_text_chat_default_server_template",
                    "token_accounting_note": "Deployment input count; model weights and tokenizer hash are not independently verified"}
                self._cache[key] = deepcopy(result)
                while len(self._cache) > MAX_CACHE_ENTRIES:
                    self._cache.popitem(last=False)
            result["tokenization_wire_sha256"] = wire_hash
            event.update(result, status="ok")
            return result
        except urllib.error.HTTPError as exc:
            event.update(http_status=exc.code, error_category="http_error")
            raise TokenizationError("http_error") from None
        except TokenizationError as exc:
            event["error_category"] = exc.category
            if exc.http_status is not None:
                event["http_status"] = exc.http_status
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            event["error_category"] = "connection_or_timeout"
            raise TokenizationError("connection_or_timeout") from None
        except Exception:
            # Provider/transport exceptions can contain request text or keys.
            event["error_category"] = "unexpected_transport_error"
            raise TokenizationError("unexpected_transport_error") from None
        finally:
            event["seconds"] = max(0.0, time.monotonic() - started)
            self.record_event(event)

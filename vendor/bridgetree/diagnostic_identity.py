"""Public deployment and canonical computation identity (never credentials)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Mapping


@dataclass(frozen=True)
class DeploymentIdentity:
    model_id: str = ""
    checkpoint_revision: str = ""
    weights_sha256: str = ""
    quantization: str = ""
    tokenizer_id: str = ""
    tokenizer_revision: str = ""
    chat_template_revision: str = ""
    serving_engine: str = ""
    serving_version: str = ""
    deployment_revision: str = ""
    identity_source: str = "unknown"
    verification_evidence_id: str = ""

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not isinstance(value, str):
                raise ValueError(f"deployment_identity.{key} must be a string")
            if len(value) > 512 or "\n" in value or "\r" in value:
                raise ValueError(f"deployment_identity.{key} is not a short identity")
            if re.search(r"https?://|Bearer\s|sk-[A-Za-z0-9_-]{8,}", value, re.I):
                raise ValueError("deployment_identity must not contain URLs or credentials")
        if self.identity_source not in {"unknown", "operator_declared", "server_reported", "verified"}:
            raise ValueError("unsupported deployment identity_source")
        if self.identity_source == "verified" and not self.verification_evidence_id:
            raise ValueError("verified identity requires verification_evidence_id")

    @classmethod
    def from_value(cls, value: Any) -> "DeploymentIdentity":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("deployment_identity must be a mapping")
        return cls(**dict(value))

    @property
    def declared(self) -> bool:
        return bool(self.weights_sha256 or self.checkpoint_revision or self.deployment_revision)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_request_payload(request: Mapping[str, Any]) -> dict[str, Any]:
    """Remove plan-only transport metadata; preserve every provider parameter."""
    payload = dict(request)
    payload.pop("endpoint", None)
    payload.pop("endpoint_sha256", None)
    return payload


def request_hash(request: Mapping[str, Any]) -> str:
    return content_hash(canonical_request_payload(request))


def deployment_fingerprint(config: Any) -> str:
    identity = DeploymentIdentity.from_value(getattr(config, "deployment_identity", {}))
    value = asdict(identity)
    value.pop("identity_source")
    value.pop("verification_evidence_id")
    if not identity.declared:
        # Missing deployment identity cannot safely share a cache across runs.
        # A task scope supplies a frozen run identity; standalone clients use
        # a process-local nonce rather than pretending an endpoint pins weights.
        from .request_audit import cache_scope_identity
        value["unverified_cache_scope"] = cache_scope_identity()
    return content_hash({"schema": 1, "deployment": value})


def deployment_public(config: Any) -> dict[str, Any]:
    identity = DeploymentIdentity.from_value(getattr(config, "deployment_identity", {}))
    return {**asdict(identity), "identity_declared": identity.declared,
            "deployment_fingerprint": deployment_fingerprint(config)}


def execution_hash(request: Mapping[str, Any], fingerprint: str) -> str:
    return content_hash({"schema": 1, "request_hash": request_hash(request), "deployment_fingerprint": fingerprint})


def validate_request_param_credentials(value: Any, *, reject_transport_metadata: bool = False) -> None:
    """Reject credential-bearing request options before a plan is persisted.

    Header containers are transport configuration, not generation payload
    parameters.  Reject them even when a nested key is not a recognized
    credential alias.  Normalize spelling, not values, so legitimate tools,
    response formats and numeric token-budget options are left unchanged.
    """
    if not isinstance(value, Mapping):
        raise ValueError("request parameters must be a mapping")
    forbidden = {
        "apikey", "apikeyenv", "xapikey", "authorization", "proxyauthorization",
        "auth", "authtoken", "bearertoken", "accesstoken", "refreshtoken",
        "clientsecret", "secretkey", "privatekey", "secretaccesskey",
        "password", "passwd", "secret", "token", "credential", "credentials",
        "header", "headers", "extraheaders", "httpheaders", "requestheaders",
        "cookie", "cookies", "setcookie",
    }
    if reject_transport_metadata:
        forbidden.update({"endpoint", "endpointsha256"})

    def visit(item):
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str) or re.sub(r"[-_\s]", "", key).lower() in forbidden:
                    raise ValueError("request parameters may not contain credentials or transport metadata")
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
    visit(value)


def validate_provider_params(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("provider_request_params must be a mapping")
    validate_request_param_credentials(value, reject_transport_metadata=True)
    reserved = {"model", "messages", "temperature", "max_tokens"}
    if reserved.intersection(value):
        raise ValueError("provider_request_params may not replace canonical generation fields")
    # JSON roundtrip also rejects NaN and copies caller-owned nested mappings.
    return json.loads(canonical_json(dict(value)))

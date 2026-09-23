"""Generate .claude/skills/reachy-api/endpoints.md from the live daemon openapi.json.

Re-run after a daemon update so the reference matches the robot:
    python tools/gen_api_reference.py [--host H] [--no-examples]

Only read-only GET endpoints listed in SAFE_EXAMPLE_GETS are called to capture example responses.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from robot.connect import resolve_host  # noqa: E402
from robot.preflight import Api, ApiError  # noqa: E402

OUTPUT = os.path.join(ROOT, ".claude", "skills", "reachy-api", "endpoints.md")

GROUPS = OrderedDict([
    ("Daemon", ("/api/daemon", "/health-check")),
    ("State", ("/api/state",)),
    ("Move", ("/api/move",)),
    ("Motors", ("/api/motors",)),
    ("Media, tracking & sounds", ("/api/media",)),
    ("Camera", ("/api/camera",)),
    ("Audio & volume", ("/api/audio", "/api/volume")),
    ("Apps", ("/api/apps",)),
    ("Kinematics", ("/api/kinematics",)),
    ("Hugging Face auth", ("/api/hf-auth",)),
    ("Wi-Fi", ("/wifi",)),
    ("Update", ("/update",)),
    ("Cache", ("/cache",)),
    ("Web pages (redirects)", ("/",)),
])

# Read-only, fast, non-sensitive GETs whose live responses are embedded as examples.
SAFE_EXAMPLE_GETS = [
    "/api/daemon/status", "/api/daemon/robot-name", "/api/daemon/hardware-id",
    "/api/daemon/robot-app-lock-status", "/api/state/full", "/api/state/present_head_pose",
    "/api/state/present_body_yaw", "/api/state/present_antenna_joint_positions", "/api/state/doa",
    "/api/motors/status", "/api/move/running", "/api/media/status", "/api/media/tracking/face",
    "/api/media/sounds", "/api/camera/specs", "/api/volume/current", "/api/volume/microphone/current",
    "/api/apps/current-app-status", "/api/apps/startup-app", "/api/apps/list-available/installed",
    "/api/kinematics/info", "/api/hf-auth/status", "/api/hf-auth/oauth/configured",
    "/wifi/status", "/wifi/error", "/update/install-source",
    "/api/move/recorded-move-datasets/list/pollen-robotics/reachy-mini-emotions-library",
]

# Require explicit user confirmation before calling (see CLAUDE.md).
DANGEROUS = {
    ("POST", "/api/daemon/start"), ("POST", "/api/daemon/stop"), ("POST", "/api/daemon/restart"),
    ("POST", "/api/daemon/robot-name"),
    ("POST", "/update/start"), ("POST", "/update/start-from-ref"),
    ("POST", "/wifi/setup_hotspot"), ("POST", "/wifi/connect"), ("POST", "/wifi/forget"),
    ("POST", "/wifi/forget_all"), ("POST", "/wifi/connect_sealed"), ("POST", "/wifi/reset_error"),
    ("POST", "/cache/clear-hf"), ("POST", "/cache/reset-apps"),
    ("POST", "/api/apps/install"), ("POST", "/api/apps/remove/{app_name}"),
    ("POST", "/api/apps/stop-current-app"), ("POST", "/api/apps/restart-current-app"),
    ("POST", "/api/apps/update/{app_name}"), ("POST", "/api/apps/install-private-space"),
    ("PUT", "/api/apps/startup-app"),
    ("POST", "/api/audio/config/apply"),
    ("POST", "/api/hf-auth/save-token"), ("DELETE", "/api/hf-auth/token"), ("POST", "/api/hf-auth/refresh-relay"),
}


def group_of(path: str) -> str:
    for name, prefixes in GROUPS.items():
        if name.startswith("Web pages"):
            continue
        if any(path.startswith(p) for p in prefixes):
            return name
    return "Web pages (redirects)"


def type_str(schema: dict) -> str:
    if not schema:
        return "any"
    if "$ref" in schema:
        return f"`{schema['$ref'].split('/')[-1]}`"
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            return " \\| ".join(type_str(s) for s in schema[key])
    if "enum" in schema:
        return " \\| ".join(f"`{e}`" for e in schema["enum"])
    t = schema.get("type", "any")
    if t == "array":
        return f"array[{type_str(schema.get('items', {}))}]"
    if t == "object" and "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
        return f"object[str → {type_str(schema['additionalProperties'])}]"
    return t


def one_line(text: str | None) -> str:
    return " ".join((text or "").split())


def render_params(op: dict) -> list[str]:
    params = op.get("parameters") or []
    if not params:
        return []
    lines = ["", "| Param | In | Required | Type | Default | Description |", "|---|---|---|---|---|---|"]
    for p in params:
        schema = p.get("schema", {})
        default = schema.get("default", "")
        lines.append(
            f"| `{p['name']}` | {p['in']} | {'yes' if p.get('required') else 'no'} | {type_str(schema)} "
            f"| {json.dumps(default) if default != '' else ''} | {one_line(p.get('description') or schema.get('description'))} |"
        )
    return lines


def render_body(op: dict) -> list[str]:
    body = op.get("requestBody")
    if not body:
        return []
    parts = []
    for ctype, content in body.get("content", {}).items():
        parts.append(f"{ctype}: {type_str(content.get('schema', {}))}")
    return ["", f"**Body** ({'required' if body.get('required') else 'optional'}): " + "; ".join(parts)]


def render_response(op: dict) -> list[str]:
    ok = (op.get("responses") or {}).get("200") or {}
    content = ok.get("content") or {}
    if not content:
        return []
    schema = next(iter(content.values())).get("schema", {})
    return ["", f"**Response 200**: {type_str(schema)}"]


def render_schema(name: str, schema: dict) -> list[str]:
    lines = [f"### `{name}`", ""]
    if schema.get("description"):
        lines += [one_line(schema["description"]), ""]
    if "enum" in schema:
        lines.append("Enum: " + ", ".join(f"`{e}`" for e in schema["enum"]))
        return lines + [""]
    props = schema.get("properties") or {}
    if not props:
        return lines + [f"Type: {type_str(schema)}", ""]
    required = set(schema.get("required") or [])
    lines += ["| Field | Type | Required | Default | Description |", "|---|---|---|---|---|"]
    for field_name, prop in props.items():
        default = json.dumps(prop["default"]) if "default" in prop else ""
        lines.append(
            f"| `{field_name}` | {type_str(prop)} | {'yes' if field_name in required else 'no'} | {default} "
            f"| {one_line(prop.get('description'))} |"
        )
    return lines + [""]


def capture_examples(host: str) -> dict[str, str]:
    api = Api(host, timeout=8.0)
    examples = {}
    for path in SAFE_EXAMPLE_GETS:
        try:
            text = json.dumps(api.get(path), indent=2)
        except (ApiError, ValueError) as exc:
            text = f"<error: {exc}>"
        if len(text) > 1500:
            text = text[:1500] + "\n... (truncated)"
        examples[path] = text
    return examples


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=None)
    parser.add_argument("--no-examples", action="store_true")
    args = parser.parse_args(argv)
    host = args.host or resolve_host()

    spec = Api(host, timeout=10.0).get("/openapi.json")
    examples = {} if args.no_examples else capture_examples(host)
    version = (Api(host).get("/api/daemon/status") or {}).get("version", "?")

    by_group: dict[str, list] = OrderedDict((g, []) for g in GROUPS)
    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            by_group[group_of(path)].append((method.upper(), path, op))

    out = [
        "# Reachy Mini daemon REST API — endpoint reference",
        "",
        f"Generated by `tools/gen_api_reference.py` from `http://{host}:8000/openapi.json` (daemon {version}).",
        "Base URL: `http://<host>:8000`. 🔴 = ask the user before calling. Example responses are live captures.",
        "",
        "## Index",
        "",
    ]
    for group, ops in by_group.items():
        if ops:
            anchor = group.lower().replace(" ", "-").replace(",", "").replace("&", "").replace("(", "").replace(")", "")
            out.append(f"- [{group}](#{anchor}) — {len(ops)} endpoints")
    out.append("")

    for group, ops in by_group.items():
        if not ops:
            continue
        out += [f"## {group}", ""]
        for method, path, op in ops:
            flag = " 🔴" if (method, path) in DANGEROUS else ""
            out += [f"### `{method} {path}`{flag}", "", f"**{op.get('summary', '')}** — {one_line(op.get('description'))}"]
            out += render_params(op) + render_body(op) + render_response(op)
            example_key = path if method == "GET" and path in examples else None
            if example_key is None and method == "GET" and "{source_kind}" in path:
                example_key = "/api/apps/list-available/installed"
            if example_key is None and method == "GET" and "{dataset_name}" in path:
                example_key = "/api/move/recorded-move-datasets/list/pollen-robotics/reachy-mini-emotions-library"
            if example_key in examples:
                out += ["", f"Example (`GET {example_key}`):", "", "```json", examples[example_key], "```"]
            out.append("")

    out += ["## Schemas", ""]
    for name, schema in spec.get("components", {}).get("schemas", {}).items():
        out += render_schema(name, schema)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {OUTPUT} ({sum(len(o) for o in by_group.values())} endpoints)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

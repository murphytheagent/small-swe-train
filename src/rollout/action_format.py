"""Shared parse/render helpers for assistant-action payloads.

This module centralizes the assistant-action surface so future payload-format
migrations can move one entrypoint at a time instead of updating each consumer
independently.
"""

from __future__ import annotations

import functools
import json
import re
import types
from dataclasses import dataclass
from html import escape
from typing import Any, Literal, Mapping, Union, get_args, get_origin, get_type_hints
from xml.etree import ElementTree as ET

from config import (
    ACTION_PARSE_MODE,
    ACTION_PAYLOAD_FORMAT,
    MAX_TOOL_CALLS_PER_TURN,
    SUPPORTED_ACTION_PARSE_MODES,
    SUPPORTED_ACTION_PAYLOAD_FORMATS,
)
from prompts.model_delimiters import ModelDelimiters, default_delimiters
from schemas import ActionEnvelope, TOOL_SCHEMAS, ToolCall, canonical_tool_name, make_tool_call

from .turn_parser import (
    TurnParseError,
    extract_chatml_assistant_payload,
    parse_assistant_turn_payload,
)

ActionPayloadFormat = Literal["json", "xml"]
ActionParseMode = Literal["json_only", "dual", "xml_only"]

_XML_TOOL_CALL_RE = re.compile(r"<tool_call\b[^>]*\bname\s*=")
_DISALLOWED_XML_SNIPPETS = ("<!doctype", "<!entity", "<?", "<!--")
_VALID_XML_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_JSON_DECODER = json.JSONDecoder()


@dataclass(frozen=True)
class ParsedAssistantAction:
    envelope: ActionEnvelope
    payload_format: ActionPayloadFormat


def _normalize_action_payload_format(value: str | None) -> ActionPayloadFormat:
    normalized = str(value or ACTION_PAYLOAD_FORMAT).strip().lower()
    if normalized not in SUPPORTED_ACTION_PAYLOAD_FORMATS:
        allowed = ", ".join(SUPPORTED_ACTION_PAYLOAD_FORMATS)
        raise ValueError(f"Unsupported action payload format {normalized!r}. Expected one of: {allowed}.")
    return normalized  # type: ignore[return-value]


def _normalize_action_parse_mode(value: str | None) -> ActionParseMode:
    normalized = str(value or ACTION_PARSE_MODE).strip().lower()
    if normalized not in SUPPORTED_ACTION_PARSE_MODES:
        allowed = ", ".join(SUPPORTED_ACTION_PARSE_MODES)
        raise ValueError(f"Unsupported action parse mode {normalized!r}. Expected one of: {allowed}.")
    return normalized  # type: ignore[return-value]


def is_chatml_assistant_turn(
    assistant_text: str,
    *,
    delimiters: ModelDelimiters | None = None,
) -> bool:
    """Return True when *assistant_text* looks like a full ChatML assistant turn."""
    d = delimiters or default_delimiters()
    return assistant_text.strip().startswith(f"{d.role_start}assistant")


def parse_assistant_text(
    assistant_text: str,
    *,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN,
    parse_mode: str | None = None,
) -> ActionEnvelope:
    """Parse raw assistant text as either ChatML turn text or bare payload."""
    return parse_assistant_text_result(
        assistant_text,
        max_tool_calls=max_tool_calls,
        parse_mode=parse_mode,
    ).envelope


def parse_assistant_text_result(
    assistant_text: str,
    *,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN,
    parse_mode: str | None = None,
) -> ParsedAssistantAction:
    """Parse raw assistant text and record which payload format succeeded."""
    stripped = assistant_text.strip()
    if is_chatml_assistant_turn(stripped):
        payload = extract_chatml_assistant_payload(stripped)
    else:
        payload = stripped
    return parse_assistant_payload(
        payload,
        max_tool_calls=max_tool_calls,
        parse_mode=parse_mode,
    )


def parse_assistant_payload(
    payload: str,
    *,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN,
    parse_mode: str | None = None,
) -> ParsedAssistantAction:
    """Parse bare assistant payload text with the configured format policy."""
    normalized_mode = _normalize_action_parse_mode(parse_mode)
    if normalized_mode == "json_only":
        return _parse_assistant_payload_as(
            payload,
            payload_format="json",
            max_tool_calls=max_tool_calls,
        )
    if normalized_mode == "xml_only":
        return _parse_assistant_payload_as(
            payload,
            payload_format="xml",
            max_tool_calls=max_tool_calls,
        )

    format_hint = _detect_payload_format_hint(payload)
    candidate_formats: tuple[ActionPayloadFormat, ...]
    if format_hint == "json":
        candidate_formats = ("json", "xml")
    elif format_hint == "xml":
        candidate_formats = ("xml", "json")
    else:
        candidate_formats = ("json", "xml")

    first_error: TurnParseError | None = None
    for candidate_format in candidate_formats:
        try:
            return _parse_assistant_payload_as(
                payload,
                payload_format=candidate_format,
                max_tool_calls=max_tool_calls,
            )
        except TurnParseError as exc:
            if first_error is None:
                first_error = exc

    assert first_error is not None
    raise first_error


def _parse_assistant_payload_as(
    payload: str,
    *,
    payload_format: ActionPayloadFormat,
    max_tool_calls: int,
) -> ParsedAssistantAction:
    if payload_format == "json":
        return ParsedAssistantAction(
            envelope=_parse_json_assistant_turn_payload_dual(
                payload,
                max_tool_calls=max_tool_calls,
            ),
            payload_format="json",
        )
    return ParsedAssistantAction(
        envelope=parse_xml_assistant_turn_payload(payload, max_tool_calls=max_tool_calls),
        payload_format="xml",
    )


def _detect_payload_format_hint(payload: str) -> ActionPayloadFormat | None:
    first_json = payload.find("<tool_call>")
    xml_match = _XML_TOOL_CALL_RE.search(payload)
    first_xml = xml_match.start() if xml_match else -1

    if first_json < 0 and first_xml < 0:
        return None
    if first_json >= 0 and (first_xml < 0 or first_json < first_xml):
        return "json"
    return "xml"


def _parse_json_assistant_turn_payload_dual(
    payload: str,
    *,
    max_tool_calls: int,
) -> ActionEnvelope:
    d = default_delimiters()
    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be >= 1")

    thinking: str | None = None
    think_seen = False
    tool_calls: list[ToolCall] = []
    cursor = 0

    while cursor < len(payload):
        think_start = payload.find(d.think_start, cursor)
        tool_start = payload.find(d.tool_call_start, cursor)
        supported_starts = [start for start in (think_start, tool_start) if start != -1]
        next_supported_start = min(supported_starts) if supported_starts else -1

        xml_match = _XML_TOOL_CALL_RE.search(payload, cursor)
        if xml_match is not None and (next_supported_start == -1 or xml_match.start() < next_supported_start):
            xml_sequence = _find_xml_tool_call_sequence(payload, xml_match.start())
            if xml_sequence is not None:
                xml_end, _ = xml_sequence
                if _looks_like_xml_tool_call_sequence(payload[xml_match.start() : xml_end]):
                    next_think_after_xml = payload.find(d.think_start, xml_end)
                    next_json_after_xml = payload.find(d.tool_call_start, xml_end)
                    boundary_candidates = [
                        start for start in (next_think_after_xml, next_json_after_xml) if start != -1
                    ]
                    boundary = min(boundary_candidates) if boundary_candidates else len(payload)
                    if (
                        not payload[cursor : xml_match.start()].strip()
                        and not payload[xml_end:boundary].strip()
                    ):
                        raise TurnParseError("Mixed JSON/XML assistant payloads are not allowed.")
                    cursor = xml_end
                    continue

        starts = [start for start in (think_start, tool_start) if start != -1]
        if not starts:
            break

        next_start = min(starts)
        if think_start != -1 and think_start == next_start and (tool_start == -1 or think_start < tool_start):
            xml_think_end = _find_xml_cdata_think_end(payload, think_start)
            if xml_think_end is not None:
                cursor = xml_think_end
                continue
            if think_seen:
                raise TurnParseError(
                    f"At most one {d.think_start} block is allowed per assistant turn."
                )
            think_end = payload.find(d.think_end, think_start + len(d.think_start))
            if think_end < 0:
                raise TurnParseError(f"Unbalanced {d.think_start} delimiters.")
            think_seen = True
            raw_thinking = payload[think_start + len(d.think_start) : think_end].strip()
            thinking = raw_thinking or None
            cursor = think_end + len(d.think_end)
            continue

        json_start = tool_start + len(d.tool_call_start)
        while json_start < len(payload) and payload[json_start].isspace():
            json_start += 1

        try:
            payload_obj, json_end = _JSON_DECODER.raw_decode(payload, json_start)
        except json.JSONDecodeError as exc:
            raise TurnParseError(f"Invalid tool_call JSON: {exc.msg}") from exc

        end_tag_start = json_end
        while end_tag_start < len(payload) and payload[end_tag_start].isspace():
            end_tag_start += 1
        if not payload.startswith(d.tool_call_end, end_tag_start):
            raise TurnParseError(
                f"Missing {d.tool_call_end} after {d.tool_call_start} JSON payload."
            )

        if not isinstance(payload_obj, dict):
            raise TurnParseError(
                f"Each {d.tool_call_start} payload must decode to a JSON object."
            )
        try:
            tool_calls.append(make_tool_call(payload_obj))
        except ValueError as exc:
            raise TurnParseError(str(exc)) from exc
        if len(tool_calls) > max_tool_calls:
            raise TurnParseError(
                f"Too many tool calls: got {len(tool_calls)}, max is {max_tool_calls}."
            )
        cursor = end_tag_start + len(d.tool_call_end)

    if not tool_calls:
        raise TurnParseError(
            f"At least one {d.tool_call_start} block is required."
        )

    try:
        return ActionEnvelope(tool_calls=tuple(tool_calls), thinking=thinking)
    except ValueError as exc:
        raise TurnParseError(str(exc)) from exc


def _looks_like_xml_tool_call_sequence(payload_segment: str) -> bool:
    candidate = payload_segment.strip()
    if not candidate:
        return False
    try:
        root = ET.fromstring(f"<assistant_payload>{candidate}</assistant_payload>")
    except ET.ParseError:
        return False
    if root.text and root.text.strip():
        return False
    for child in root:
        if _local_xml_name(child.tag) != "tool_call":
            return False
        if child.tail and child.tail.strip():
            return False
        name = str(child.attrib.get("name", "")).strip()
        if not name:
            return False
    return True


def _find_xml_tool_call_sequence(payload: str, start: int) -> tuple[int, int] | None:
    cursor = start
    block_count = 0
    while cursor < len(payload):
        if not _XML_TOOL_CALL_RE.match(payload, cursor):
            return None if cursor == start else (cursor, block_count)
        element_end = _find_xml_element_end(payload, cursor)
        if element_end is None:
            return None
        block_count += 1
        cursor = element_end
        while cursor < len(payload) and payload[cursor].isspace():
            cursor += 1
        if not _XML_TOOL_CALL_RE.match(payload, cursor):
            return cursor, block_count
    return cursor, block_count


def _find_xml_element_end(payload: str, start: int) -> int | None:
    if start < 0 or start >= len(payload) or payload[start] != "<":
        return None

    cursor = start
    depth = 0
    while cursor < len(payload):
        if payload.startswith("<![CDATA[", cursor):
            cdata_end = payload.find("]]>", cursor + len("<![CDATA["))
            if cdata_end < 0:
                return None
            cursor = cdata_end + len("]]>")
            continue
        if payload.startswith("<!--", cursor):
            comment_end = payload.find("-->", cursor + len("<!--"))
            if comment_end < 0:
                return None
            cursor = comment_end + len("-->")
            continue

        if payload[cursor] != "<":
            cursor += 1
            continue

        tag_end = _find_xml_tag_end(payload, cursor)
        if tag_end is None:
            return None
        tag_body = payload[cursor + 1 : tag_end - 1].strip()
        if not tag_body or tag_body.startswith("?") or tag_body.startswith("!"):
            return None

        if tag_body.startswith("/"):
            depth -= 1
            if depth == 0:
                return tag_end
            if depth < 0:
                return None
        else:
            depth += 1
            if tag_body.endswith("/"):
                depth -= 1
                if depth == 0:
                    return tag_end
        cursor = tag_end
    return None


def _find_xml_tag_end(payload: str, start: int) -> int | None:
    quote_char: str | None = None
    cursor = start + 1
    while cursor < len(payload):
        char = payload[cursor]
        if quote_char is None:
            if char in {'"', "'"}:
                quote_char = char
            elif char == ">":
                return cursor + 1
        elif char == quote_char:
            quote_char = None
        cursor += 1
    return None


def _find_xml_cdata_think_end(payload: str, start: int) -> int | None:
    if not payload.startswith("<think><![CDATA[", start):
        return None
    return _find_xml_element_end(payload, start)


def serialize_tool_call_payload(
    call: ToolCall | Mapping[str, Any],
    *,
    compact: bool = False,
) -> str:
    """Serialize one tool-call payload deterministically."""
    payload = call.to_dict() if isinstance(call, ToolCall) else dict(call)
    dump_kwargs: dict[str, Any] = {
        "ensure_ascii": True,
        "sort_keys": True,
    }
    if compact:
        dump_kwargs["separators"] = (",", ":")
    return json.dumps(payload, **dump_kwargs)


def render_tool_call_block(
    call: ToolCall | Mapping[str, Any],
    *,
    delimiters: ModelDelimiters | None = None,
    payload_format: str | None = None,
    fallback_payload_format: str | None = None,
    compact: bool = False,
) -> str:
    """Render one tool-call block using the current delimiter contract."""
    resolved_format = _normalize_action_payload_format(payload_format)
    d = delimiters or default_delimiters()
    try:
        if resolved_format == "json":
            payload = serialize_tool_call_payload(call, compact=compact)
            return f"{d.tool_call_start}{payload}{d.tool_call_end}"
        return _render_xml_tool_call_block(call)
    except ValueError:
        if fallback_payload_format is None:
            raise
        resolved_fallback = _normalize_action_payload_format(fallback_payload_format)
        if resolved_fallback == resolved_format:
            raise
        return render_tool_call_block(
            call,
            delimiters=d,
            payload_format=resolved_fallback,
            compact=compact,
        )


def render_assistant_action_text(
    envelope: ActionEnvelope,
    *,
    delimiters: ModelDelimiters | None = None,
    payload_format: str | None = None,
    compact: bool = False,
) -> str:
    """Render one assistant action envelope using the current delimiter contract."""
    d = delimiters or default_delimiters()
    chunks: list[str] = []
    if envelope.thinking:
        chunks.append(f"{d.think_start}{envelope.thinking}{d.think_end}")
    chunks.extend(
        render_tool_call_block(
            call,
            delimiters=d,
            payload_format=payload_format,
            compact=compact,
        )
        for call in envelope.tool_calls
    )
    return "".join(chunks)


def parse_xml_assistant_turn_payload(
    payload: str,
    *,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN,
) -> ActionEnvelope:
    """Parse XML assistant payload into canonical action envelope."""
    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be >= 1")

    _reject_disallowed_xml_constructs(payload)
    try:
        root = ET.fromstring(f"<assistant_payload>{payload}</assistant_payload>")
    except ET.ParseError as exc:
        raise TurnParseError(f"Invalid XML assistant payload: {exc.msg}") from exc

    if root.text and root.text.strip():
        raise TurnParseError("Unexpected raw text outside XML assistant blocks.")

    thinking: str | None = None
    think_seen = False
    tool_calls: list[ToolCall] = []

    for child in root:
        tag = _local_xml_name(child.tag)
        if tag == "think":
            if think_seen:
                raise TurnParseError("At most one <think> block is allowed per assistant turn.")
            if list(child):
                raise TurnParseError("<think> blocks may not contain nested XML elements.")
            think_seen = True
            raw_thinking = child.text or ""
            thinking = raw_thinking.strip() or None
            if child.tail and child.tail.strip():
                raise TurnParseError("Unexpected raw text outside XML assistant blocks.")
            continue
        if tag != "tool_call":
            raise TurnParseError(f"Unsupported XML assistant payload tag <{tag}>.")

        tool_calls.append(_parse_xml_tool_call(child))
        if len(tool_calls) > max_tool_calls:
            raise TurnParseError(
                f"Too many tool calls: got {len(tool_calls)}, max is {max_tool_calls}."
            )
        if child.tail and child.tail.strip():
            raise TurnParseError("Unexpected raw text outside XML assistant blocks.")

    if not tool_calls:
        raise TurnParseError("At least one <tool_call ...> block is required.")

    try:
        return ActionEnvelope(tool_calls=tuple(tool_calls), thinking=thinking)
    except ValueError as exc:
        raise TurnParseError(str(exc)) from exc


def _reject_disallowed_xml_constructs(payload: str) -> None:
    cursor = 0
    while cursor < len(payload):
        cdata_start = payload.find("<![CDATA[", cursor)
        if cdata_start < 0:
            segment = payload[cursor:]
            cursor = len(payload)
        else:
            segment = payload[cursor:cdata_start]
            cdata_end = payload.find("]]>", cdata_start + len("<![CDATA["))
            if cdata_end < 0:
                raise TurnParseError("Unterminated CDATA section in XML assistant payload.")
            cursor = cdata_end + len("]]>")

        lowered = segment.lower()
        for snippet in _DISALLOWED_XML_SNIPPETS:
            if snippet in lowered:
                raise TurnParseError(
                    "XML assistant payloads do not allow comments, DTDs, entities, processing instructions, or namespaces."
                )


def _local_xml_name(tag: str) -> str:
    if "}" in tag or ":" in tag:
        raise TurnParseError("Namespaces are not allowed in assistant XML payloads.")
    return tag


def _parse_xml_tool_call(element: ET.Element) -> ToolCall:
    for attribute_name in element.attrib:
        _local_xml_name(attribute_name)

    name = str(element.attrib.get("name", "")).strip()
    if not name:
        raise TurnParseError("Each XML <tool_call> requires a non-empty name attribute.")
    if set(element.attrib) != {"name"}:
        extras = sorted(name for name in element.attrib if name != "name")
        raise TurnParseError(
            f"Unsupported attributes on XML <tool_call>: {', '.join(extras)}."
        )

    try:
        canonical_name = canonical_tool_name(name)
    except ValueError as exc:
        raise TurnParseError(str(exc)) from exc

    if element.text and element.text.strip():
        raise TurnParseError(
            f'XML <tool_call name="{canonical_name}"> may not contain raw text; use arg child elements.'
        )

    seen_fields: set[str] = set()
    args: dict[str, Any] = {}
    annotations = _tool_arg_annotations(canonical_name)
    for child in element:
        field_name = _local_xml_name(child.tag)
        for attribute_name in child.attrib:
            _local_xml_name(attribute_name)
        if child.attrib:
            extras = ", ".join(sorted(str(name) for name in child.attrib))
            raise TurnParseError(f"Unsupported attributes on XML arg <{field_name}>: {extras}.")
        if field_name in seen_fields:
            raise TurnParseError(f"Duplicate XML arg field <{field_name}> in <tool_call name=\"{canonical_name}\">.")
        seen_fields.add(field_name)
        annotation = annotations.get(field_name)
        args[field_name] = _coerce_xml_arg_value(
            tool_name=canonical_name,
            field_name=field_name,
            annotation=annotation,
            element=child,
        )
        if child.tail and child.tail.strip():
            raise TurnParseError(
                f'Unexpected raw text inside <tool_call name="{canonical_name}"> after <{field_name}>.'
            )

    try:
        return make_tool_call({"tool": canonical_name, "args": args})
    except ValueError as exc:
        raise TurnParseError(str(exc)) from exc


@functools.lru_cache(maxsize=None)
def _tool_arg_annotations(tool_name: str) -> dict[str, Any]:
    schema = TOOL_SCHEMAS.get(tool_name)
    source = schema.get("source") if isinstance(schema, Mapping) else None
    if not isinstance(source, type):
        return {}
    return dict(get_type_hints(source))


@functools.lru_cache(maxsize=None)
def _tool_arg_order(tool_name: str) -> tuple[str, ...]:
    schema = TOOL_SCHEMAS.get(tool_name)
    source = schema.get("source") if isinstance(schema, Mapping) else None
    annotations = getattr(source, "__annotations__", {})
    if isinstance(annotations, Mapping):
        return tuple(str(name) for name in annotations)
    return ()


def _coerce_xml_arg_value(
    *,
    tool_name: str,
    field_name: str,
    annotation: Any,
    element: ET.Element,
) -> Any:
    normalized_annotation = _unwrap_scalar_annotation(annotation)
    origin = get_origin(normalized_annotation)
    if origin in (list, tuple, set):
        item_tag = _xml_list_item_tag(field_name)
        values: list[str] = []
        if element.text and element.text.strip():
            raise TurnParseError(f"XML list arg <{field_name}> may not contain raw text before child elements.")
        for child in element:
            if _local_xml_name(child.tag) != item_tag:
                raise TurnParseError(
                    f"XML list arg <{field_name}> must contain repeated <{item_tag}> children."
                )
            for attribute_name in child.attrib:
                _local_xml_name(attribute_name)
            if child.attrib:
                extras = ", ".join(sorted(str(name) for name in child.attrib))
                raise TurnParseError(
                    f"Unsupported attributes on XML list item <{item_tag}> in <{field_name}>: {extras}."
                )
            if list(child):
                raise TurnParseError(f"XML list item <{item_tag}> may not contain nested elements.")
            values.append(child.text or "")
            if child.tail and child.tail.strip():
                raise TurnParseError(
                    f"Unexpected raw text inside XML list arg <{field_name}> after <{item_tag}>."
                )
        return values

    if list(element):
        raise TurnParseError(
            f"XML scalar arg <{field_name}> may not contain nested elements."
        )

    raw_text = element.text or ""
    if normalized_annotation is int:
        stripped = raw_text.strip()
        if not stripped:
            raise TurnParseError(
                f"XML integer arg <{field_name}> in <tool_call name=\"{tool_name}\"> may not be empty."
            )
        try:
            return int(stripped)
        except ValueError as exc:
            raise TurnParseError(
                f"XML integer arg <{field_name}> in <tool_call name=\"{tool_name}\"> must be an integer."
            ) from exc
    if normalized_annotation is float:
        stripped = raw_text.strip()
        if not stripped:
            raise TurnParseError(
                f"XML numeric arg <{field_name}> in <tool_call name=\"{tool_name}\"> may not be empty."
            )
        try:
            return float(stripped)
        except ValueError as exc:
            raise TurnParseError(
                f"XML numeric arg <{field_name}> in <tool_call name=\"{tool_name}\"> must be numeric."
            ) from exc
    if normalized_annotation is bool:
        stripped = raw_text.strip().lower()
        if stripped in {"true", "1"}:
            return True
        if stripped in {"false", "0"}:
            return False
        raise TurnParseError(
            f"XML boolean arg <{field_name}> in <tool_call name=\"{tool_name}\"> must be true/false."
        )
    return raw_text


def _xml_list_item_tag(field_name: str) -> str:
    if field_name == "changed_paths":
        return "path"
    return "item"


def _unwrap_scalar_annotation(annotation: Any) -> Any:
    origin = get_origin(annotation)
    union_type = getattr(types, "UnionType", None)
    if origin in tuple(item for item in (union_type, Union) if item is not None):
        args = tuple(arg for arg in get_args(annotation) if arg is not type(None))
        if len(args) == 1:
            return args[0]
    return annotation


def _render_xml_tool_call_block(call: ToolCall | Mapping[str, Any]) -> str:
    tool_name, args = _tool_name_and_args(call)
    ordered_items = _ordered_arg_items(tool_name, args)
    parts = [f'<tool_call name="{escape(tool_name, quote=True)}">']
    for field_name, value in ordered_items:
        parts.append(_render_xml_arg(field_name, value))
    parts.append("</tool_call>")
    return "".join(parts)


def _tool_name_and_args(call: ToolCall | Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    if isinstance(call, ToolCall):
        return call.tool, dict(call.args)
    tool_name = str(call.get("tool", "")).strip()
    args = call.get("args", {})
    if not tool_name:
        raise ValueError("Tool-call payload must include non-empty 'tool'.")
    if not isinstance(args, Mapping):
        raise ValueError("Tool-call payload must include mapping 'args'.")
    return tool_name, dict(args)


def _ordered_arg_items(tool_name: str, args: Mapping[str, Any]) -> list[tuple[str, Any]]:
    try:
        canonical_name = canonical_tool_name(tool_name)
    except ValueError:
        canonical_name = tool_name

    annotations = _tool_arg_annotations(canonical_name)
    if annotations:
        unknown_fields = sorted(field_name for field_name in args if field_name not in annotations)
        if unknown_fields:
            raise ValueError(
                f"Unknown XML args for tool {canonical_name!r}: {', '.join(unknown_fields)}"
            )

    ordered_names = _tool_arg_order(canonical_name)
    ordered: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for field_name in ordered_names:
        if field_name in args and args[field_name] is not None:
            ordered.append((field_name, args[field_name]))
            seen.add(field_name)
    for field_name in sorted(args):
        if field_name not in seen and args[field_name] is not None:
            ordered.append((field_name, args[field_name]))
    return ordered


def _render_xml_arg(field_name: str, value: Any) -> str:
    if not _VALID_XML_NAME_RE.fullmatch(field_name):
        raise ValueError(f"Invalid XML arg field name: {field_name!r}")
    if isinstance(value, list):
        item_tag = _xml_list_item_tag(field_name)
        items = "".join(
            f"<{item_tag}>{_wrap_cdata(str(item))}</{item_tag}>"
            for item in value
        )
        return f"<{field_name}>{items}</{field_name}>"
    if isinstance(value, bool):
        rendered_value = str(value).lower()
        return f"<{field_name}>{rendered_value}</{field_name}>"
    if isinstance(value, (int, float)):
        return f"<{field_name}>{value}</{field_name}>"
    return f"<{field_name}>{_wrap_cdata(str(value))}</{field_name}>"


def _wrap_cdata(text: str) -> str:
    return f"<![CDATA[{text.replace(']]>', ']]]]><![CDATA[>')}]]>"

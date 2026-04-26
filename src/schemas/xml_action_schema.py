"""Derived XML encoding schema for assistant tool calls.

``TOOL_SCHEMAS`` remains the authority for canonical tool names, argument
types, required fields, and constraints.  This module describes how that
canonical ``ToolCall(tool, args)`` shape is represented as model-visible XML.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Union, get_args, get_origin, get_type_hints

from .contracts import ALLOWED_TOOLS, TOOL_SCHEMAS, canonical_tool_name

XML_TOOL_CALL_ELEMENT = "tool_call"
XML_THINK_ELEMENT = "think"
XML_TOOL_NAME_ATTRIBUTE = "name"
XML_CDATA_END = "]]>"
XML_BUILTIN_ESCAPES: tuple[str, ...] = ("&lt;", "&gt;", "&amp;", "&quot;", "&apos;")

XmlStringEncoding = Literal["cdata", "escaped_text"]
XmlArgKind = Literal["string", "int", "float", "bool", "list[string]"]


@dataclass(frozen=True)
class XmlArgSchema:
    name: str
    kind: XmlArgKind
    required: bool
    string_encoding: XmlStringEncoding | None = None
    list_item_tag: str | None = None
    constraints: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class XmlToolSchema:
    tool: str
    element: str
    name_attribute: str
    args: tuple[XmlArgSchema, ...]

    @property
    def arg_names(self) -> tuple[str, ...]:
        return tuple(arg.name for arg in self.args)


def get_xml_tool_schema(tool_name: str) -> XmlToolSchema:
    canonical = canonical_tool_name(tool_name)
    schema = TOOL_SCHEMAS.get(canonical)
    if not isinstance(schema, Mapping):
        return XmlToolSchema(
            tool=canonical,
            element=XML_TOOL_CALL_ELEMENT,
            name_attribute=XML_TOOL_NAME_ATTRIBUTE,
            args=(),
        )

    hints = _tool_arg_hints(schema)
    required = _required_fields(schema)
    constraints = schema.get("constraints")
    constraints_by_field = dict(constraints) if isinstance(constraints, Mapping) else {}
    args: list[XmlArgSchema] = []
    for field_name, annotation in hints.items():
        kind = _xml_arg_kind(annotation)
        field_constraints = constraints_by_field.get(field_name)
        args.append(
            XmlArgSchema(
                name=field_name,
                kind=kind,
                required=field_name in required,
                string_encoding=_string_encoding(annotation),
                list_item_tag=_xml_list_item_tag(field_name) if kind == "list[string]" else None,
                constraints=MappingProxyType(
                    dict(field_constraints) if isinstance(field_constraints, Mapping) else {}
                ),
            )
        )
    return XmlToolSchema(
        tool=canonical,
        element=XML_TOOL_CALL_ELEMENT,
        name_attribute=XML_TOOL_NAME_ATTRIBUTE,
        args=tuple(args),
    )


def iter_xml_tool_schemas() -> tuple[XmlToolSchema, ...]:
    return tuple(get_xml_tool_schema(tool_name) for tool_name in ALLOWED_TOOLS)


def get_xml_arg_schema(tool_name: str, field_name: str) -> XmlArgSchema | None:
    for arg_schema in get_xml_tool_schema(tool_name).args:
        if arg_schema.name == field_name:
            return arg_schema
    return None


def get_xml_arg_order(tool_name: str) -> tuple[str, ...]:
    return get_xml_tool_schema(tool_name).arg_names


def get_xml_list_item_tag(field_name: str) -> str:
    return _xml_list_item_tag(field_name)


def render_xml_contract_signature() -> str:
    return '<tool_call name="tool_name"><arg_name><![CDATA[value]]></arg_name></tool_call>'


def render_xml_tool_schema_line(schema: XmlToolSchema) -> str:
    required_args = [arg for arg in schema.args if arg.required]
    optional_args = [arg for arg in schema.args if not arg.required]
    required = ", ".join(_xml_arg_label(arg) for arg in required_args) if required_args else "-"
    optional = ", ".join(_xml_arg_label(arg) for arg in optional_args) if optional_args else "-"
    return f"   - {schema.tool} XML args: required {{{required}}}; optional {{{optional}}}"


def render_xml_arg_placeholder(arg: XmlArgSchema) -> str:
    if arg.kind == "list[string]":
        item_tag = arg.list_item_tag or _xml_list_item_tag(arg.name)
        return f"<{arg.name}><{item_tag}><![CDATA[string]]></{item_tag}></{arg.name}>"
    if arg.kind == "string":
        return f"<{arg.name}><![CDATA[string]]></{arg.name}>"
    return f"<{arg.name}>{arg.kind}</{arg.name}>"


def _tool_arg_hints(schema: Mapping[str, Any]) -> dict[str, Any]:
    source = schema.get("source")
    if not isinstance(source, type):
        return {}
    return dict(get_type_hints(source))


def _required_fields(schema: Mapping[str, Any]) -> set[str]:
    required_raw = schema.get("required")
    if not isinstance(required_raw, (list, tuple, set)):
        return set()
    return {str(name) for name in required_raw if isinstance(name, str)}


def _xml_arg_kind(annotation: Any) -> XmlArgKind:
    normalized = _unwrap_optional(annotation)
    origin = get_origin(normalized)
    if origin in (list, tuple, set):
        return "list[string]"
    if normalized is int:
        return "int"
    if normalized is float:
        return "float"
    if normalized is bool:
        return "bool"
    return "string"


def _string_encoding(annotation: Any) -> XmlStringEncoding | None:
    kind = _xml_arg_kind(annotation)
    if kind == "string":
        return "cdata"
    return None


def _unwrap_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    union_type = getattr(types, "UnionType", None)
    if origin in tuple(item for item in (union_type, Union) if item is not None):
        args = tuple(arg for arg in get_args(annotation) if arg is not type(None))
        if len(args) == 1:
            return args[0]
    return annotation


def _xml_list_item_tag(field_name: str) -> str:
    if field_name == "changed_paths":
        return "path"
    return "item"


def _xml_arg_label(arg: XmlArgSchema) -> str:
    qualifiers: list[str] = []
    min_length = arg.constraints.get("min_length")
    minimum = arg.constraints.get("minimum")
    maximum = arg.constraints.get("maximum")
    if min_length is not None:
        qualifiers.append(f"min_len={min_length}")
    if minimum is not None and maximum is not None:
        qualifiers.append(f"{minimum}..{maximum}")
    elif minimum is not None:
        qualifiers.append(f">={minimum}")
    elif maximum is not None:
        qualifiers.append(f"<={maximum}")
    if arg.kind == "string":
        qualifiers.append("cdata_or_escaped_text")
    if arg.kind == "list[string]":
        item_tag = arg.list_item_tag or _xml_list_item_tag(arg.name)
        qualifiers.append(f"items=<{item_tag}>")
    suffix = f"({', '.join(qualifiers)})" if qualifiers else ""
    return f"{arg.name}:{arg.kind}{suffix}"

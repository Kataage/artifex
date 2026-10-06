from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from typing import Any


class WorkflowSourceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SourceWorkflowInfo:
    workflow_id: str
    sha256: str
    node_count: int
    api_node_count: int
    output_node_id: str


class UiWorkflowCompiler:
    """Compile a ComfyUI UI workflow into the backend API prompt shape.

    Only node-declared input names are serialized, so UI-only widget state such as
    control_after_generate is never sent to the backend. Reroutes are collapsed,
    muted nodes become absent inputs, bypassed nodes forward a type-compatible input,
    and only ancestors of the selected output node are retained.
    """

    REROUTE_TYPES = frozenset({"Reroute (rgthree)"})
    MODE_NEVER = 2
    MODE_BYPASS = 4

    def __init__(
        self,
        source: Mapping[str, Any],
        *,
        mode_overrides: Mapping[int, int] | None = None,
    ) -> None:
        self._source = copy.deepcopy(dict(source))
        raw_nodes = self._source.get("nodes")
        raw_links = self._source.get("links")
        if not isinstance(raw_nodes, list) or not isinstance(raw_links, list):
            raise WorkflowSourceError("ComfyUI source must contain nodes and links")
        self._nodes: dict[int, dict[str, Any]] = {
            int(node["id"]): node
            for node in raw_nodes
            if isinstance(node, dict) and "id" in node
        }
        self._links: dict[int, list[Any]] = {
            int(link[0]): link
            for link in raw_links
            if isinstance(link, list) and len(link) >= 6
        }
        self._mode_overrides = dict(mode_overrides or {})

    def set_input_link(self, node_id: int, input_name: str, link_id: int | None) -> None:
        node = self._require_node(node_id)
        for item in node.get("inputs", ()):
            if item.get("name") == input_name:
                item["link"] = link_id
                return
        raise WorkflowSourceError(f"source input not found: {node_id}.{input_name}")

    def set_widget(self, node_id: int, input_name: str, value: Any) -> None:
        node = self._require_node(node_id)
        named = node.setdefault("widgets_values_named", {})
        if not isinstance(named, dict):
            raise WorkflowSourceError(
                f"source node {node_id} has invalid widgets_values_named"
            )
        named[input_name] = value

    def compile(self, output_node_id: int) -> tuple[dict[str, dict[str, Any]], SourceWorkflowInfo]:
        if output_node_id not in self._nodes:
            raise WorkflowSourceError(f"output node is missing: {output_node_id}")

        ancestors = self._ancestors(output_node_id)
        graph: dict[str, dict[str, Any]] = {}
        for node_id in sorted(ancestors):
            node = self._nodes[node_id]
            if not self._is_serialized_node(node_id, node):
                continue
            inputs: dict[str, Any] = {}
            named = node.get("widgets_values_named")
            named_values = named if isinstance(named, dict) else {}
            for item in node.get("inputs", ()):
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if not isinstance(name, str) or not name:
                    continue
                link_id = item.get("link")
                if isinstance(link_id, int):
                    resolved = self._resolve_link(link_id)
                    if resolved is not None:
                        inputs[name] = [str(resolved[0]), resolved[1]]
                    continue
                if name in named_values:
                    inputs[name] = copy.deepcopy(named_values[name])

            api_node: dict[str, Any] = {
                "class_type": str(node.get("type", "")),
                "inputs": inputs,
            }
            title = node.get("title")
            if isinstance(title, str) and title:
                api_node["_meta"] = {"title": title}
            graph[str(node_id)] = api_node

        raw = json.dumps(self._source, ensure_ascii=False, sort_keys=True).encode("utf-8")
        workflow_id = str(self._source.get("id", "unknown"))
        return graph, SourceWorkflowInfo(
            workflow_id=workflow_id,
            sha256=hashlib.sha256(raw).hexdigest(),
            node_count=len(self._nodes),
            api_node_count=len(graph),
            output_node_id=str(output_node_id),
        )

    def _ancestors(self, start_id: int) -> set[int]:
        seen: set[int] = set()
        stack = [start_id]
        while stack:
            node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            node = self._nodes[node_id]
            for item in node.get("inputs", ()):
                if not isinstance(item, dict):
                    continue
                link_id = item.get("link")
                if not isinstance(link_id, int):
                    continue
                resolved = self._resolve_link(link_id)
                if resolved is not None:
                    stack.append(resolved[0])
        return seen

    def _resolve_link(self, link_id: int) -> tuple[int, int] | None:
        link = self._links.get(link_id)
        if link is None:
            raise WorkflowSourceError(f"source link is missing: {link_id}")
        return self._resolve_output(int(link[1]), int(link[2]), set())

    def _resolve_output(
        self,
        node_id: int,
        output_slot: int,
        seen: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        key = (node_id, output_slot)
        if key in seen:
            raise WorkflowSourceError(f"cycle while resolving source output: {key}")
        seen.add(key)

        node = self._require_node(node_id)
        mode = self._mode(node_id, node)
        if mode == self.MODE_NEVER:
            return None

        node_type = str(node.get("type", ""))
        if node_type in self.REROUTE_TYPES:
            inputs = node.get("inputs", ())
            if not inputs:
                return None
            link_id = inputs[0].get("link") if isinstance(inputs[0], dict) else None
            if not isinstance(link_id, int):
                return None
            link = self._links.get(link_id)
            if link is None:
                return None
            return self._resolve_output(int(link[1]), int(link[2]), seen)

        if mode == self.MODE_BYPASS:
            outputs = node.get("outputs", ())
            output_type = None
            if 0 <= output_slot < len(outputs) and isinstance(outputs[output_slot], dict):
                output_type = outputs[output_slot].get("type")
            candidates = [
                item
                for item in node.get("inputs", ())
                if isinstance(item, dict)
                and isinstance(item.get("link"), int)
                and (output_type is None or item.get("type") == output_type)
            ]
            if not candidates:
                candidates = [
                    item
                    for item in node.get("inputs", ())
                    if isinstance(item, dict) and isinstance(item.get("link"), int)
                ]
            if not candidates:
                return None
            chosen = candidates[0]
            link = self._links.get(int(chosen["link"]))
            if link is None:
                return None
            return self._resolve_output(int(link[1]), int(link[2]), seen)

        return node_id, output_slot

    def _is_serialized_node(self, node_id: int, node: Mapping[str, Any]) -> bool:
        mode = self._mode(node_id, node)
        return (
            mode not in {self.MODE_NEVER, self.MODE_BYPASS}
            and str(node.get("type", "")) not in self.REROUTE_TYPES
        )

    def _mode(self, node_id: int, node: Mapping[str, Any]) -> int:
        raw = self._mode_overrides.get(node_id, node.get("mode", 0))
        return int(raw) if isinstance(raw, int | float) else 0

    def _require_node(self, node_id: int) -> dict[str, Any]:
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise WorkflowSourceError(f"source node is missing: {node_id}") from exc


def load_illust_main_source() -> dict[str, Any]:
    root = files("artifex.comfy").joinpath("workflow_sources")
    raw = root.joinpath("illust_main_2026_10.ui.json").read_text(encoding="utf-8")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise WorkflowSourceError("illust Main source must be a JSON object")
    return value

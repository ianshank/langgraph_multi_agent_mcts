"""Bulk registration helper for builtin tools."""

from __future__ import annotations

from pathlib import Path

from src.framework.harness.settings import HarnessPermissions
from src.framework.harness.tools.builtins.fs import file_edit_hashed_tool, file_read_tool
from src.framework.harness.tools.builtins.shell import (
    lint_run_tool,
    shell_tool,
    test_run_tool,
    type_check_run_tool,
)
from src.framework.harness.tools.registry import ToolRegistry


def register_builtin_tools(
    registry: ToolRegistry,
    *,
    root: Path,
    perms: HarnessPermissions,
    correlation_id: str | None = None,
    shell_allowlist: list[str] | None = None,
) -> None:
    """Register every builtin tool against ``registry``.

    Permissions on individual tools still apply: a permissionless registry
    will surface 'permission denied' messages instead of executing.
    """

    schema, handler = file_read_tool(root=root, perms=perms)
    registry.register(schema, handler)

    schema, handler = file_edit_hashed_tool(root=root, perms=perms)
    registry.register(schema, handler)

    schema, handler = shell_tool(cwd=root, perms=perms, correlation_id=correlation_id, allowlist=shell_allowlist)
    registry.register(schema, handler)

    schema, handler = test_run_tool(cwd=root, perms=perms, correlation_id=correlation_id)
    registry.register(schema, handler)

    schema, handler = lint_run_tool(cwd=root, perms=perms, correlation_id=correlation_id)
    registry.register(schema, handler)

    schema, handler = type_check_run_tool(cwd=root, perms=perms, correlation_id=correlation_id)
    registry.register(schema, handler)


__all__ = ["register_builtin_tools"]

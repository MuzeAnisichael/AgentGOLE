"""Built-in tools that read and write files below one directory."""

from pathlib import Path

from ..errors import ConfigError, ToolError
from ..protocols import Tool
from .function import tool


def files(root: str | Path, write: bool = False) -> list[Tool]:
    """Create ``read_file`` and ``list_dir``, plus ``write_file`` when ``write=True``.

    Every path is resolved relative to ``root`` and must stay inside it.

    Example:
        gole.Agent(tools=[gole.tools.files("./notes")])
    """
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise ConfigError(f"File tools need an existing directory as root: {root}")

    def locate(path: str) -> Path:
        if Path(path).is_absolute():
            raise ToolError("Use a path relative to the root directory.")
        # Resolving follows symbolic links, so a link that points outside is caught here.
        target = (base / path).resolve()
        if target != base and base not in target.parents:
            raise ToolError("The path leaves the root directory.")
        return target

    def read_file(path: str) -> str:
        """Read a text file.

        Args:
            path: File path relative to the root directory.
        """
        return locate(path).read_text(encoding="utf-8", errors="replace")

    def list_dir(path: str = ".") -> str:
        """List the entries of a directory; directory names end with a slash.

        Args:
            path: Directory path relative to the root directory.
        """
        entries = sorted(locate(path).iterdir(), key=lambda entry: entry.name)
        return "\n".join(entry.name + ("/" if entry.is_dir() else "") for entry in entries)

    def write_file(path: str, content: str) -> str:
        """Create or replace a text file, creating parent directories as needed.

        Args:
            path: File path relative to the root directory.
            content: Complete new content of the file.
        """
        target = locate(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {target.relative_to(base).as_posix()}."

    tools = [tool(read_file), tool(list_dir)]
    return [*tools, tool(write_file)] if write else tools

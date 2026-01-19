"""systemd_mcp package entry point stubs."""

from importlib.metadata import version, PackageNotFoundError


def __getattr__(name: str):
    if name == "__version__":
        try:
            return version("systemd_mcp")
        except PackageNotFoundError as exc:
            raise AttributeError("Package version unavailable") from exc
    raise AttributeError(name)


__all__ = ["__version__"]

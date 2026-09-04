"""Public experiment-manager entry point with corrected CLI flag semantics.

The optimization implementation is kept in ``_run_experiment_manager_impl``.
This facade preserves the documented public API while ensuring that
``--no-figures`` actually disables layout figure rendering.
"""

from __future__ import annotations

from typing import Any

from whl_experiments import _run_experiment_manager_impl as _impl


_ORIGINAL_BUILD_PARSER = _impl.build_parser


def build_parser():
    """Return the public parser with conventional ``--no-figures`` semantics."""
    parser = _ORIGINAL_BUILD_PARSER()
    for action in parser._actions:
        if action.dest == "no_figures":
            action.const = True
            action.default = False
            action.help = "Disable layout figure rendering."
            break
    return parser


for _name in _impl.__all__:
    if _name != "build_parser":
        globals()[_name] = getattr(_impl, _name)

__all__ = list(_impl.__all__)
_impl.build_parser = build_parser


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


def main() -> None:
    _impl.main()


if __name__ == "__main__":
    main()

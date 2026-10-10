"""Explicit method capabilities shared by execution, transport and reporting."""
from dataclasses import dataclass

LOCAL = 'dagbt_local_terminal_v1'
UNSCORED = 'dagbt_local_terminal_proxy_free_v1'
JOINT = 'dagbt_joint_memory_v1'
RESIDUAL = 'dagbt_residual_memory_v1'


@dataclass(frozen=True)
class Capabilities:
    terminal: bool = False
    scoring: bool = True
    joint_reading: bool = False
    residual_control: bool = False


_CAPABILITIES = {
    LOCAL: Capabilities(terminal=True),
    UNSCORED: Capabilities(terminal=True, scoring=False),
    JOINT: Capabilities(terminal=True, scoring=False, joint_reading=True),
    RESIDUAL: Capabilities(terminal=True, scoring=False, joint_reading=True, residual_control=True),
}


def capabilities(version):
    return _CAPABILITIES.get(version, Capabilities())


def terminal_method(version):
    return capabilities(version).terminal

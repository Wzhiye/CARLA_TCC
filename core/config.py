"""Configuration loading.

The demo is configured with two YAML files (``config/sim.yaml`` and
``config/fleet.yaml``) that are merged into one nested dict. Scenario files
live in ``scenarios/`` and are loaded by the scenario package.
"""

import copy
import os

try:
    import yaml
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise SystemExit(
        "PyYAML is missing. Inside myproject run:  uv pip install pyyaml"
    ) from exc


PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)
DEFAULT_SIM_CONFIG = os.path.join(PACKAGE_DIR, "config", "sim.yaml")
DEFAULT_FLEET_CONFIG = os.path.join(PACKAGE_DIR, "config", "fleet.yaml")


def load_yaml(path):
    """Load one YAML file and always return a dict."""
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def deep_merge(base, override):
    """Return a new dict where ``override`` is merged recursively into ``base``."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def get(cfg, dotted_key, default=None):
    """Read a nested value with a dotted path, e.g. ``get(cfg, "world.map")``."""
    node = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def resolve_path(path):
    """Resolve a path from the config relative to the ra_demo package folder."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(PACKAGE_DIR, path))


def load_config(sim_path=None, fleet_path=None, overrides=None):
    """Load and merge the simulation and fleet configuration files."""
    cfg = load_yaml(sim_path or DEFAULT_SIM_CONFIG)
    cfg = deep_merge(cfg, load_yaml(fleet_path or DEFAULT_FLEET_CONFIG))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg

"""Load and validate scenario YAML files.

A scenario either places actors (props, stalled vehicles) or raises an event
on a robotaxi (``event:`` block, no actors). Actors are placed

* at fixed sites on the map (``trigger.type: site`` + ``sites:``): any
  robotaxi that drives there meets the incident, or
* on the target robotaxi's route (``trigger.type: route_ahead``), which is
  also what the operator's inject keys (T / F1..F5) use for every scenario.
"""

import glob
import logging
import os

from ra_demo.core.config import deep_merge, load_yaml

LOG = logging.getLogger(__name__)

TRIGGER_TYPES = ("site", "route_ahead", "event", "manual")
ACTIVATIONS = ("approach", "always")
PLACEMENT_MODES = ("lane", "junction_exit")
PATTERNS = ("single", "lane_block", "vehicle", "road_block")
# "any": no requirement; "driving": a driving lane in either direction.
SIDE_REQUIREMENTS = ("any", "driving", "driving_same", "driving_opposite")

DEFAULTS = {
    "enabled": True,
    "target_vehicle": "RT-01",
    "trigger": {
        "type": "route_ahead",
        "auto": True,              # False: only injected by the operator (T / F-keys)
        "start_after_s": 20.0,
        "min_distance_m": 50.0,
        "max_distance_m": 120.0,
        "retry_interval_s": 1.0,
        "give_up_after_s": 120.0,
        # site only
        "activation": "approach",  # approach: spawn when a robotaxi heads there | always: keep it there
        "activate_within_m": 150.0,  # a robotaxi whose route reaches the site this close activates it
        "approach_min_m": 50.0,    # ...unless it is already closer (no props popping up in front of it)
        # event only
        "interval_s": None,        # [min, max]: random time between events (seeded)
        "require_passenger": True, # target: random -> only cars driving a passenger
    },
    "sites": {},                   # {map name: [{name, road_id, lane_id, s} | {name, location: [x, y]}]}
    "placement": {
        "mode": "lane",            # lane | junction_exit
        "zone_length_m": 16.0,
        "clear_before_m": 15.0,
        "clear_after_m": 20.0,
        "left_lane": "any",
        "right_lane": "any",
        "require_empty_lane": True,
        # junction_exit only
        "after_junction_m": 4.0,
        "route_through": "straight",   # straight | any
        "require_turn": "any",         # any | left | right | both
        # site only: robotaxis routed through a site pass these points in the closed lane
        "via_before_m": 25.0,
        "via_after_m": 8.0,
    },
    "lane_guard": {
        "enabled": True,
        "upstream_m": 30.0,
        "remove_within_m": 8.0,
        "open_side": None,             # side traffic is nudged to; default from the actors
    },
    "actors": [],
    "event": None,
    "cleanup": {
        "after_passed_s": 8.0,
        "max_active_s": 600.0,
    },
    "repeat": {
        "enabled": False,
        "rearm_after_s": 60.0,
    },
}


class ScenarioSpecError(ValueError):
    pass


def validate(spec, source):
    if "id" not in spec:
        raise ScenarioSpecError("{}: missing 'id'".format(source))
    trigger_type = spec["trigger"]["type"]
    if trigger_type not in TRIGGER_TYPES:
        raise ScenarioSpecError("{}: unknown trigger type '{}'".format(source, trigger_type))
    if trigger_type == "event":
        event = spec.get("event") or {}
        if "reason" not in event:
            raise ScenarioSpecError("{}: an event scenario needs event.reason".format(source))
        interval = spec["trigger"].get("interval_s")
        if interval is not None and (len(interval) != 2 or float(interval[0]) > float(interval[1])):
            raise ScenarioSpecError("{}: trigger.interval_s must be [min, max]".format(source))
        return spec
    if spec["trigger"]["activation"] not in ACTIVATIONS:
        raise ScenarioSpecError("{}: trigger.activation must be one of {}".format(
            source, ", ".join(ACTIVATIONS)))
    if not isinstance(spec.get("sites") or {}, dict):
        raise ScenarioSpecError("{}: 'sites' must map a map name to a list of sites".format(source))
    if spec["placement"]["mode"] not in PLACEMENT_MODES:
        raise ScenarioSpecError("{}: placement.mode must be one of {}".format(
            source, ", ".join(PLACEMENT_MODES)))
    for side in ("left_lane", "right_lane"):
        value = spec["placement"][side]
        if value not in SIDE_REQUIREMENTS:
            raise ScenarioSpecError("{}: placement.{} must be one of {}".format(
                source, side, ", ".join(SIDE_REQUIREMENTS)))
    if not spec["actors"]:
        raise ScenarioSpecError("{}: 'actors' is empty".format(source))
    for index, actor in enumerate(spec["actors"]):
        if "blueprint" not in actor:
            raise ScenarioSpecError("{}: actors[{}] has no 'blueprint'".format(source, index))
        if actor.get("pattern", "single") not in PATTERNS:
            raise ScenarioSpecError("{}: actors[{}] has unknown pattern '{}'".format(
                source, index, actor.get("pattern")))
    return spec


def load_scenarios(directory, enabled_ids=None):
    """Load every ``*.yaml`` in ``directory``; keep only ``enabled_ids`` if given (in that order)."""
    specs = []
    for path in sorted(glob.glob(os.path.join(directory, "*.yaml"))):
        spec = validate(deep_merge(DEFAULTS, load_yaml(path)), os.path.basename(path))
        if enabled_ids is not None and spec["id"] not in enabled_ids:
            continue
        if not spec.get("enabled", True):
            continue
        spec["_source"] = path
        specs.append(spec)
        LOG.info("Loaded scenario %s (%s)", spec["id"], os.path.basename(path))
    if enabled_ids is not None:
        order = {scenario_id: index for index, scenario_id in enumerate(enabled_ids)}
        specs.sort(key=lambda s: order.get(s["id"], len(order)))
    return specs

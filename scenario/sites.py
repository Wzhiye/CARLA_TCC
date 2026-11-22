"""Fixed incident sites: places on the map where a scenario happens.

A site is anchored by OpenDRIVE coordinates (``road_id``, ``lane_id``, ``s``),
which ``python -m ra_demo.tools.pick_point`` prints for any spot on the map.
Any robotaxi whose route runs through a site meets the incident there; the
scenario manager spawns the props when one is approaching (or keeps them in
the world all the time with ``activation: always``).
"""

import logging

import carla

from ra_demo.core import compat, geometry
from ra_demo.scenario import patterns
from ra_demo.scenario.placement import Anchor, _lane_continuous

LOG = logging.getLogger(__name__)

# Robotaxi routes are sampled every 2 m; widen the s-range a little so a
# waypoint just before/after the zone edge still counts as "in the zone".
_S_TOLERANCE = 1.0


def _lane_point_before(waypoint, distance, step=1.0):
    """A point up to ``distance`` metres back along the same lane (None if there is no room)."""
    current, travelled = waypoint, 0.0
    while travelled + step <= distance:
        previous = current.previous(step)
        if not previous or previous[0].is_junction or previous[0].road_id != waypoint.road_id \
                or previous[0].lane_id != waypoint.lane_id:
            break
        current = previous[0]
        travelled += step
    return current if travelled >= 5.0 else None


def _map_key(map_name):
    """'Carla/Maps/Town10HD_Opt' -> 'town10hd' (the _Opt variant shares the road network)."""
    name = compat.short_map_name(map_name).lower()
    return name[:-4] if name.endswith("_opt") else name


def sites_for_map(spec, map_name):
    """The site list of ``spec`` for this map ([] if the map has none)."""
    wanted = _map_key(map_name)
    for key, sites in (spec.get("sites") or {}).items():
        if _map_key(key) == wanted:
            return list(sites or [])
    return []


class Site:
    """One resolved site: lane geometry, the s-range it closes and a via point."""

    def __init__(self, scenario_id, name, waypoint, spec):
        placement = spec["placement"]
        self.scenario_id = scenario_id
        self.name = name
        self.waypoint = waypoint
        self.zone_length = float(placement["zone_length_m"])
        self.road_id = waypoint.road_id
        self.closed_lane_ids = {lane.lane_id for lane in patterns.closed_lanes(waypoint, spec["actors"])}
        end = waypoint.next(self.zone_length)
        s_end = end[0].s if end else waypoint.s
        self.s_lo, self.s_hi = sorted((waypoint.s, s_end))
        # Robotaxis sent "through" the site are routed via a point in the closed lane
        # before the zone and one after it, so their route runs through the zone and any
        # lane change into that lane happens well before the zone, not inside it.
        via = waypoint.next(self.zone_length + float(placement.get("via_after_m", 8.0)))
        self.via_location = (via[0] if via else waypoint).transform.location
        self.via_locations = [self.via_location]
        before = _lane_point_before(waypoint, float(placement.get("via_before_m", 25.0)))
        if before is not None:
            self.via_locations.insert(0, before.transform.location)
        self.anchor = Anchor(waypoint, 0.0, self.zone_length, waypoint.lane_width,
                             compat.describe_side_lane(waypoint, "left"),
                             compat.describe_side_lane(waypoint, "right"))

    @property
    def location(self):
        return self.waypoint.transform.location

    def covers(self, waypoint):
        """True if a route waypoint lies in the closed part of the road."""
        return (waypoint.road_id == self.road_id and waypoint.lane_id in self.closed_lane_ids
                and self.s_lo - _S_TOLERANCE <= waypoint.s <= self.s_hi + _S_TOLERANCE)

    def route_distance(self, vehicle_location, plan, limit=None):
        """Distance along ``plan`` from the vehicle to the zone, or None if the plan
        does not run through the site (within ``limit`` metres)."""
        total = 0.0
        previous = vehicle_location
        for waypoint, _ in plan:
            location = waypoint.transform.location
            total += geometry.planar_distance(previous, location)
            previous = location
            if limit is not None and total > limit:
                return None
            if self.covers(waypoint):
                return total
        return None

    def describe(self):
        return "{} road {} lane {} s {:.1f}".format(self.name, self.road_id, self.waypoint.lane_id,
                                                     self.waypoint.s)


def resolve_sites(spec, world_map):
    """Turn the YAML site entries for the current map into :class:`Site` objects."""
    resolved = []
    placement = spec["placement"]
    needed = float(placement["zone_length_m"]) + float(placement.get("clear_after_m", 0.0))
    for index, entry in enumerate(sites_for_map(spec, world_map.name)):
        name = str(entry.get("name") or "{}#{}".format(spec["id"], index + 1))
        waypoint = None
        if "road_id" in entry:
            waypoint = world_map.get_waypoint_xodr(int(entry["road_id"]), int(entry["lane_id"]),
                                                   float(entry.get("s", 0.0)))
        elif "location" in entry:
            x, y = entry["location"][:2]
            z = entry["location"][2] if len(entry["location"]) > 2 else 0.0
            waypoint = world_map.get_waypoint(carla.Location(float(x), float(y), float(z)))
        if waypoint is None:
            LOG.warning("Scenario %s: site %s does not exist on %s", spec["id"], name, world_map.name)
            continue
        if waypoint.lane_type != carla.LaneType.Driving or waypoint.is_junction:
            LOG.warning("Scenario %s: site %s is not on a driving lane outside a junction",
                        spec["id"], name)
            continue
        if not _lane_continuous(waypoint, needed):
            LOG.warning("Scenario %s: site %s needs %.0f m of straight lane after it", spec["id"],
                        name, needed)
        site = Site(spec["id"], name, waypoint, spec)
        resolved.append(site)
        LOG.info("Scenario %s: site %s", spec["id"], site.describe())
    return resolved

"""Ride missions: an endless loop of pickups and drop-offs.

Destinations are drawn from the map's spawn points (they always lie on a
driving lane). A fixed list of spawn-point indices can be configured per
vehicle to make a demo route fully repeatable.

With fixed incident sites on the map, a share of the trips
(``via_site_probability``) is routed through one of them, so the demo does not
depend on robotaxis passing the sites by chance.
"""

import random

from ra_demo.core import geometry

PHASE_TO_PICKUP = "TO_PICKUP"
PHASE_TO_DROPOFF = "TO_DROPOFF"


class RideMission:
    """Produces the next destination for one robotaxi."""

    def __init__(self, spawn_points, cfg, seed, fixed_stops=None):
        self._points = spawn_points
        self._rng = random.Random(seed)
        self._min_distance = float(cfg.get("min_trip_distance_m", 120.0))
        self._fixed = list(fixed_stops or [])
        self._fixed_index = 0
        self.phase = PHASE_TO_DROPOFF   # flips to TO_PICKUP for the first trip
        self.trips_completed = 0
        self.destination_index = None
        self.via_probability = float(cfg.get("via_site_probability", 0.5))
        self.last_site = None

    def next_destination(self, current_location, avoid=(), avoid_radius=50.0):
        """Pick the next stop and flip the ride phase. Returns (index, transform).

        Random stops within ``avoid_radius`` of a location in ``avoid`` (incident
        sites) are skipped: a drop-off right behind a work zone leaves no room
        to steer around it.
        """
        self.phase = PHASE_TO_PICKUP if self.phase == PHASE_TO_DROPOFF else PHASE_TO_DROPOFF
        if self._fixed:
            index = self._fixed[self._fixed_index % len(self._fixed)]
            self._fixed_index += 1
        else:
            index = self._random_index(current_location, avoid, avoid_radius)
        self.destination_index = index
        return index, self._points[index]

    def choose_site(self, offers):
        """Pick a site to drive through on this trip, or None.

        ``offers`` is [(site, state, number of robotaxis already heading there)].
        Sites nobody is heading to are preferred, and the site of the previous
        detour only when it is the only one.
        """
        if not offers or self._rng.random() >= self.via_probability:
            return None
        fewest = min(claims for _, _, claims in offers)
        candidates = [site for site, _, claims in offers if claims == fewest]
        if len(candidates) > 1 and self.last_site in candidates:
            candidates.remove(self.last_site)
        return self._rng.choice(candidates)

    def _random_index(self, current_location, avoid=(), avoid_radius=50.0):
        candidates = [
            i for i, point in enumerate(self._points)
            if geometry.planar_distance(point.location, current_location) >= self._min_distance
            and all(geometry.planar_distance(point.location, spot) >= avoid_radius for spot in avoid)
        ]
        if not candidates:
            candidates = list(range(len(self._points)))
        return self._rng.choice(candidates)

    def arrived(self):
        if self.phase == PHASE_TO_DROPOFF:
            self.trips_completed += 1

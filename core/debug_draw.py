"""World-space debug drawing (visible in the server window and in camera images).

Three modes, cycled with D in the viewer:

* ``subtle`` (default): a few small, dim dots along each robotaxi's route and
  along an active maneuver path. Nothing else.
* ``full``: thin route lines, perception corridors, scenario zones and labels.
* ``off``: nothing.
"""

import carla

MODES = ("subtle", "full", "off")

COLORS = {
    "route_dot": carla.Color(45, 75, 105),
    "route_line": carla.Color(60, 110, 160),
    "path_dot": carla.Color(120, 90, 30),
    "path_line": carla.Color(170, 120, 40),
    "corridor": carla.Color(40, 120, 80),
    "obstacle": carla.Color(200, 50, 50),
    "zone": carla.Color(170, 110, 20),
    "text": carla.Color(200, 200, 200),
    "request": carla.Color(200, 60, 200),
}


class DebugDrawer:
    """Small wrapper around ``world.debug`` with display modes.

    Every draw call is a request to the server, so shapes are redrawn only every
    ``every_n_ticks`` ticks and live just long enough to bridge the gap.
    """

    def __init__(self, world, fixed_delta, mode="subtle", every_n_ticks=4):
        self._debug = world.debug
        self.mode = mode if mode in MODES else "subtle"
        self._every = max(1, int(every_n_ticks))
        self._counter = 0
        self._life = fixed_delta * self._every * 1.2

    @property
    def enabled(self):
        return self.mode != "off"

    @property
    def full(self):
        return self.mode == "full"

    def cycle(self):
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        return self.mode

    def due(self):
        """True on the ticks where shapes should be redrawn."""
        if not self.enabled:
            return False
        self._counter = (self._counter + 1) % self._every
        return self._counter == 0

    # ------------------------------------------------------------- shapes
    def route(self, plan, max_points=15, z=0.15):
        """The planned route: sparse dim dots (subtle) or a thin line (full)."""
        if not self.enabled:
            return
        previous = None
        for index, (waypoint, _) in enumerate(plan):
            if index >= max_points:
                break
            location = waypoint.transform.location + carla.Location(z=z)
            if self.full:
                if previous is not None:
                    self._debug.draw_line(previous, location, thickness=0.03,
                                          color=COLORS["route_line"], life_time=self._life)
                previous = location
            elif index % 3 == 2:
                self._debug.draw_point(location, size=0.05, color=COLORS["route_dot"],
                                       life_time=self._life)

    def path(self, points, z):
        """A maneuver path given as world (x, y) points."""
        if not self.enabled or not points:
            return
        step = 1 if self.full else 3
        previous = None
        for x, y in points[::step]:
            location = carla.Location(x, y, z + 0.15)
            if self.full and previous is not None:
                self._debug.draw_line(previous, location, thickness=0.04,
                                      color=COLORS["path_line"], life_time=self._life)
            elif not self.full:
                self._debug.draw_point(location, size=0.06, color=COLORS["path_dot"],
                                       life_time=self._life)
            previous = location

    def polygon(self, polygon, z, color="corridor", thickness=0.03):
        if not self.full or polygon is None or polygon.is_empty:
            return
        # Buffered corridors have many round-join vertices; a coarse outline is enough.
        coords = list(polygon.simplify(0.3).exterior.coords)
        for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
            self._debug.draw_line(carla.Location(x0, y0, z), carla.Location(x1, y1, z),
                                  thickness=thickness, color=COLORS[color], life_time=self._life)

    def actor_box(self, actor, color="obstacle"):
        if not self.full:
            return
        transform = actor.get_transform()
        local = actor.bounding_box.location
        # Transform.transform() returns a Vector3D; BoundingBox needs a Location.
        center = transform.transform(carla.Location(local.x, local.y, local.z))
        box = carla.BoundingBox(carla.Location(center.x, center.y, center.z),
                                actor.bounding_box.extent)
        self._debug.draw_box(box, transform.rotation, thickness=0.05,
                             color=COLORS[color], life_time=self._life)

    def text(self, location, text, color="text", z=2.5):
        if not self.full:
            return
        self._debug.draw_string(location + carla.Location(z=z), text, draw_shadow=True,
                                color=COLORS[color], life_time=self._life)

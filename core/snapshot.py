"""Per-tick actor snapshot shared by every module.

Fetching the actor list once per tick (instead of once per agent and per check)
keeps the client loop cheap even with several agents.
"""

import carla

_LEFT = int(carla.VehicleLightState.LeftBlinker)
_RIGHT = int(carla.VehicleLightState.RightBlinker)


def has_hazard_lights(vehicle):
    """True if both blinkers are on. Asks the server, so it raises RuntimeError for a
    vehicle that was destroyed since the actor list was fetched."""
    state = int(vehicle.get_light_state())
    return bool(state & _LEFT) and bool(state & _RIGHT)


class ActorSnapshot:
    """Lists of the relevant actors for the current tick."""

    _TRAFFIC_LIGHT_REFRESH_TICKS = 200
    _cached_lights = None
    _lights_age = 0

    def __init__(self, world, frame, sim_time, fleet_ids=()):
        self.frame = frame
        self.sim_time = sim_time
        actors = world.get_actors()
        self.vehicles = list(actors.filter("vehicle.*"))
        self.walkers = list(actors.filter("walker.pedestrian.*"))
        self.props = list(actors.filter("static.prop.*"))

        # Stalled vehicles: not ours, standing still, hazard lights on. They are
        # treated like static obstacles (a broken-down van, a double-parked truck).
        # The actor list can still contain a vehicle the server has just destroyed
        # (Traffic Manager or the lane guard removed it); such vehicles are dropped.
        self.stalled = {}
        gone = set()
        for vehicle in self.vehicles:
            if vehicle.id in fleet_ids or vehicle.get_velocity().length() > 0.1:
                continue
            try:
                if has_hazard_lights(vehicle):
                    self.stalled[vehicle.id] = vehicle
            except RuntimeError:
                gone.add(vehicle.id)
        if gone:
            self.vehicles = [vehicle for vehicle in self.vehicles if vehicle.id not in gone]
        self.vehicles_by_id = {vehicle.id: vehicle for vehicle in self.vehicles}
        self.props_by_id = {prop.id: prop for prop in self.props}

        # Traffic lights never change during a run; refresh them rarely.
        cls = ActorSnapshot
        if cls._cached_lights is None or cls._lights_age >= cls._TRAFFIC_LIGHT_REFRESH_TICKS:
            cls._cached_lights = list(actors.filter("traffic.traffic_light*"))
            cls._lights_age = 0
        cls._lights_age += 1
        self.traffic_lights = cls._cached_lights

    def actor(self, actor_id):
        """A prop or vehicle of this snapshot by id (None if gone)."""
        return self.props_by_id.get(actor_id) or self.vehicles_by_id.get(actor_id)

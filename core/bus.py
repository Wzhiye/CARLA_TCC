"""Remote-assistance message bus.

Vehicles publish *requests* (vehicle-initiated, with the options they consider
feasible); the operator console sends *commands*; everything is appended to a
JSON-lines event log for later KPI analysis.

v1 is an in-process object. A later WebSocket bridge only needs to forward
these calls, so the vehicle and console code does not change.
"""

import collections
import itertools
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field

LOG = logging.getLogger(__name__)

# Reason codes a vehicle can raise.
REASON_LANE_BLOCKED = "LANE_BLOCKED"
REASON_STOPPED_VEHICLE = "STOPPED_VEHICLE"
REASON_PEDESTRIAN = "PEDESTRIAN_IN_PATH"
REASON_ROAD_CLOSED = "ROAD_CLOSED"
REASON_PASSENGER = "PASSENGER_ASSISTANCE"
REASON_SENSOR = "SENSOR_DEGRADED"
REASON_MANEUVER_FAILED = "MANEUVER_FAILED"
REASON_MANEUVER_ABORTED = "MANEUVER_ABORTED"

# Operator actions.
ACTION_BYPASS_LEFT = "BYPASS_LEFT"
ACTION_BYPASS_RIGHT = "BYPASS_RIGHT"
ACTION_TURN_LEFT = "TURN_LEFT"
ACTION_TURN_RIGHT = "TURN_RIGHT"
ACTION_PULL_OVER = "PULL_OVER"
ACTION_RESUME = "RESUME"
ACTION_ABORT = "ABORT"
ALL_ACTIONS = (ACTION_BYPASS_LEFT, ACTION_BYPASS_RIGHT, ACTION_TURN_LEFT, ACTION_TURN_RIGHT,
               ACTION_PULL_OVER, ACTION_RESUME, ACTION_ABORT)


@dataclass
class RARequest:
    request_id: str
    vehicle: str
    reason: str
    created_sim_time: float
    location: tuple
    options: list
    detail: dict = field(default_factory=dict)
    status: str = "OPEN"          # OPEN | RESOLVED | CANCELLED
    closed_sim_time: float = None
    resolution: str = None

    def waiting_time(self, sim_time):
        end = self.closed_sim_time if self.closed_sim_time is not None else sim_time
        return max(0.0, end - self.created_sim_time)


@dataclass
class RACommand:
    command_id: str
    vehicle: str
    action: str
    request_id: str
    sim_time: float


class RABus:
    """In-process request/command exchange with a JSONL event log."""

    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, "ra_events_{}.jsonl".format(stamp))
        self._log = open(self.log_path, "a", encoding="utf-8")
        self._ids = itertools.count(1)
        self._requests = collections.OrderedDict()
        self._commands = collections.defaultdict(collections.deque)
        self.notices = collections.deque(maxlen=8)

    # --------------------------------------------------------------- requests
    def open_request(self, vehicle, reason, sim_time, location, options, detail=None):
        request = RARequest(
            request_id="R{:04d}".format(next(self._ids)),
            vehicle=vehicle,
            reason=reason,
            created_sim_time=sim_time,
            location=tuple(round(value, 2) for value in location),
            options=list(options),
            detail=detail or {},
        )
        self._requests[request.request_id] = request
        self._write("request_opened", asdict(request))
        self.notice("{} requests assistance: {}".format(vehicle, reason))
        LOG.info("[RA] %s %s opened by %s (options: %s)", request.request_id, reason,
                 vehicle, ", ".join(options) or "-")
        return request

    def close_request(self, request_id, status, sim_time, resolution):
        request = self._requests.get(request_id)
        if request is None or request.status != "OPEN":
            return
        request.status = status
        request.closed_sim_time = sim_time
        request.resolution = resolution
        self._write("request_closed", {
            "request_id": request_id, "vehicle": request.vehicle, "status": status,
            "resolution": resolution, "waited_s": round(request.waiting_time(sim_time), 2),
        })
        LOG.info("[RA] %s %s (%s) after %.1fs", request_id, status, resolution,
                 request.waiting_time(sim_time))

    def open_requests(self):
        return [r for r in self._requests.values() if r.status == "OPEN"]

    def open_request_for(self, vehicle):
        for request in self._requests.values():
            if request.vehicle == vehicle and request.status == "OPEN":
                return request
        return None

    def all_requests(self):
        return list(self._requests.values())

    # --------------------------------------------------------------- commands
    def send_command(self, vehicle, action, sim_time, request_id=None):
        command = RACommand("C{:04d}".format(next(self._ids)), vehicle, action, request_id, sim_time)
        self._commands[vehicle].append(command)
        self._write("command_sent", asdict(command))
        return command

    def take_commands(self, vehicle):
        queue = self._commands[vehicle]
        commands = list(queue)
        queue.clear()
        return commands

    def reply(self, command, accepted, sim_time, note=""):
        self._write("command_reply", {
            "command_id": command.command_id, "vehicle": command.vehicle,
            "action": command.action, "accepted": accepted, "note": note, "sim_time": sim_time,
        })
        self.notice("{} {} {}{}".format(command.vehicle, "accepted" if accepted else "rejected",
                                        command.action, ": " + note if note else ""))

    # ------------------------------------------------------------------ misc
    def notice(self, text):
        self.notices.appendleft(text)

    def log_event(self, kind, payload):
        self._write(kind, payload)

    def _write(self, kind, payload):
        record = {"wall_time": round(time.time(), 3), "event": kind}
        record.update(payload)
        self._log.write(json.dumps(record, default=str) + "\n")
        self._log.flush()

    def close(self):
        if not self._log.closed:
            self._log.close()

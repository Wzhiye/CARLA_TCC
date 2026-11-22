# Robotaxi Remote-Assistance Demo (CARLA 0.9.16)

Milestones implemented: **M0** (skeleton, synchronous loop, tools),
**M1** (3 robotaxis on a ride loop, background traffic, viewer),
**M4** (operator commands: bypass left/right, turn left/right, pull over,
resume, abort; the options offered always match what the car would accept)
and scenarios **S1-S5**. Incidents happen at **fixed places on the map**: any
robotaxi that drives there meets them. Passenger calls come at random times.

| Id | Situation | Request | Expected answer | Where / when (Town10HD) |
|---|---|---|---|---|
| S1_construction | cones + sign + barrier close the lane, left side open | `LANE_BLOCKED` | `BYPASS_LEFT` | sites S1@A (avenue, road 10), S1@B (divided road, road 20) |
| S2_stalled_vehicle | broken-down van with hazard lights, slightly to the right | `STOPPED_VEHICLE` (after 5 s of waiting) | `BYPASS_LEFT` | S2@A (two-lane street: pass in the oncoming lane), S2@B (road 5) |
| S3_road_closed | barriers close the road right after a junction | `ROAD_CLOSED` (car stops before the junction) | the `TURN_*` that is green | S3@A (behind junction 189), S3@B (behind junction 23) |
| S4_passenger_assist | passenger presses the help button (no obstacle) | `PASSENGER_ASSISTANCE` (car drives on at 20 km/h) | `PULL_OVER`, later `RESUME` | a random robotaxi with a passenger, every 90-240 s |
| S5_construction_right | mirrored S1: lane closed, only the right side open | `LANE_BLOCKED` | `BYPASS_RIGHT` | S5@A (avenue, road 10, other direction) |

Every scenario can also be injected by hand on the selected robotaxi (see below).

## Fixed incident sites

Each scenario YAML lists its sites per map, in OpenDRIVE coordinates:

```yaml
trigger:
  type: site
  activation: approach       # props appear when a robotaxi's route reaches the site...
  activate_within_m: 150     # ...within 150 m (out of camera range)
sites:
  Town10HD_Opt:
    - {name: A, road_id: 10, lane_id: -2, s: 60.0}
```

* The props are placed when a robotaxi whose route runs through the site is
  150 m away, and removed once no robotaxi is heading there any more; the site
  comes back after `repeat.rearm_after_s`. With `activation: always` the props
  stay on the map all the time.
* Every robotaxi that drives there meets the incident, one after the other
  (a second car queues behind the first and asks for help once it is first).
* So that the demo does not depend on chance, half of the trips are routed
  through one of the sites (`mission.via_site_probability` in `fleet.yaml`;
  0 = only by chance, 1 = every trip). Pickups and drop-offs are never placed
  right at a site.
* To add a site: fly there with `python -m ra_demo.tools.pick_point`, press
  Enter, copy `road_id`, `lane_id` and `s` into the YAML. Choose a straight lane
  with 15 m free before the zone and 20 m after it; for S3, the first metres of
  a road behind a junction that has a detour. On a map with no sites listed,
  a scenario only runs when injected (`T` / F-keys).
* With debug drawing on `full` (`D`), every armed site is labelled in the world.

## Setup (once)

From the `myproject` folder (the uv venv already has carla 0.9.16):

```bat
uv pip install pyyaml
```

## Run

1. Start the server: `CARLA_0.9.16\CarlaUE4.exe` (add `-quality-level=Low` while developing).
2. From `myproject`:

```bat
.venv\Scripts\python -m ra_demo
```

The first run loads `Town10HD_Opt` if another map is running.

Useful options:

```bat
.venv\Scripts\python -m ra_demo --no-traffic                 :: fleet + scenario only
.venv\Scripts\python -m ra_demo --map Town03_Opt --seed 7
.venv\Scripts\python -m ra_demo --no-viewer --duration 1800 --auto-clear 30   :: 30 min soak test
```

### Operator console (pygame window)

The window shows the selected robotaxi's **front view** with three **rear-view
mirrors** (rear centre on top, left and right at the bottom corners; mirrored
images at 10 Hz). `V` switches front / chase / top view, `M` hides the mirrors.

Every action is a button in the bar at the bottom (click it) and has a key.
Buttons are green only when the selected robotaxi would accept that action right
now. A grey button shows the reason when you hover over it or press it (e.g.
"not enough room on the right: needs a 3.3 m shift, the shoulder lane allows
2.5 m"), and the side panel lists the reasons for the open request. The car
re-checks the options about once a second while a request is open, so a button
can turn green once the oncoming lane clears.

| Key | Action |
|---|---|
| `1` `2` `3` / `TAB` | select robotaxi / jump to the next open RA request |
| `<-` / `->` | bypass the obstacle on the left / right |
| `Q` / `E` | turn left / right at the next junction (re-route) |
| `P` | pull over (or secure in place when there is no room) |
| `R` | resume (leave the curb, or continue after an abort) |
| `X` | abort the running maneuver: stop and hold |
| `ENTER` | confirm a pending maneuver (or press/click the same action again) |
| `T` | inject the chosen scenario on the **selected** robotaxi |
| `Y` | choose which scenario `T` injects (S1 -> S5 -> S1 ...) |
| `F1`..`F5` | inject S1..S5 directly on the selected robotaxi (order = `scenarios.enabled` in `sim.yaml`) |
| `C` / `Shift+C` | clear the incident blocking the selected robotaxi / all scenarios |
| `V` / `M` | camera front/chase/top / rear-view mirrors on/off |
| `F` | the server window's camera follows the selection, with the same view as this window (front by default; the mirrors exist only in this window) |
| `D` | debug drawing: subtle (small dim route dots) -> full -> off |
| `H` / `ESC` | help / cancel a pending command, otherwise quit |

Maneuver commands need a confirmation (`viewer.confirm_commands` in `sim.yaml`).
If letter keys do nothing, switch the keyboard to English (an active Chinese IME
swallows letters); the buttons always work.

## Remote-assistance commands

The robotaxi asks for help (`LANE_BLOCKED`, `STOPPED_VEHICLE`, `PEDESTRIAN_IN_PATH`,
`ROAD_CLOSED`, `PASSENGER_ASSISTANCE`, `MANEUVER_FAILED`, `MANEUVER_ABORTED`) and
lists the options it would accept. Each option is a dry run of the same check
the car runs when the command arrives, so offered options are never rejected
for geometry reasons (traffic can still change between offer and command). The
car still checks every command itself and may reject it with a reason (shown
in the side panel and logged).

| Command | What the vehicle does | Typical rejections |
|---|---|---|
| `BYPASS_LEFT/RIGHT` | Plans a smooth lateral shift around the whole obstacle cluster (cones, sign, barrier, stalled car) keeping 0.7 m clearance, waits for a gap in the target lane, drives past at 10 km/h and returns to its lane. On an opposite-direction lane it waits until no oncoming car can arrive while it is out of its lane, and keeps checking until it commits | not blocked; no usable lane on that side; not enough room (needed vs. available shift); too close to steer around; the bypass would run into a junction; the obstacle is beyond the junction (turn instead) |
| `TURN_LEFT/RIGHT` | Re-routes through the next junction in that direction, then on to the destination (avoiding the blocked road) | no junction ahead; obstacle is before the junction; no such turn; no route to the destination after that turn |
| `PULL_OVER` | Moves to the rightmost lane and eases onto the shoulder/curb, stops with hazard lights; right in front of an obstacle it secures the car in place | inside a junction; no space before the next junction |
| `RESUME` | Leaves the curb, or continues an aborted bypass from where it stopped, then the trip | nothing to resume |
| `ABORT` | Cancels the running maneuver, holds position, re-opens a request | nothing to abort |

`C` removes what blocks the selected car: its scenario props, stray props from
an earlier run, or a stalled background car ("towed"). On start-up the demo
also removes props and cars left behind by a crashed run
(`world.cleanup_leftovers`).

Background (Traffic Manager) cars cannot see props either. While a scenario is
active, cars in the closed lane are asked to change lanes towards the open side;
one that is still in the lane 8 m (S3: 10 m) before the props is removed. The
stalled van in S2 is a real vehicle with the hand brake set; robotaxis treat any
stopped non-fleet car with hazard lights as a static obstacle and stop 10 m
behind it (`driving.stalled_vehicle_standoff_m`), 6 m behind props.

## Scenario authoring tools

The tools never tick the world. Run them in a second terminal.

```bat
.venv\Scripts\python -m ra_demo.tools.inspect_world info
.venv\Scripts\python -m ra_demo.tools.inspect_world props cone
.venv\Scripts\python -m ra_demo.tools.pick_point
```

`pick_point`: fly the server window camera (right mouse button + WASD), press
Enter to capture the nearest lane (road/lane id, s, width, side lanes, nearest
spawn point). `s` labels spawn-point indices, `l` labels lanes. Captures are
saved to `ra_demo/tools/picked_points.yaml`.

## Layout

```
myproject/
├─ agents/                 unmodified CARLA 0.9.16 agents (not shipped in the pip wheel)
├─ ra_demo/
│  ├─ app.py               main loop (the only place that ticks the world)
│  ├─ config/              sim.yaml, fleet.yaml
│  ├─ scenarios/           S1..S5 YAML files (one scenario each)
│  ├─ core/                world/sync/pacing, compat layer, snapshot, RA bus, geometry, debug draw
│  ├─ av/                  RobotaxiAgent (BasicAgent subclass), privileged perception, maneuvers
│  ├─ fleet/               fleet manager, ride missions, robotaxi state machine
│  ├─ traffic/             background traffic (from generate_traffic_realtime.py)
│  ├─ scenario/            loader, fixed sites, route-relative placement, prop patterns, lifecycle, lane guard
│  ├─ console/             pygame operator console
│  └─ tools/               pick_point, inspect_world
├─ tests/test_offline.py   runs without a server
└─ logs/                   RA event logs
```

Offline tests (no CARLA server needed):

```bat
.venv\Scripts\python -m unittest discover -s tests -v
```

## Acceptance checks for this round

- M0: the demo starts, runs at ~20 Hz real time, and ESC/Ctrl+C leaves the server in asynchronous mode with no leftover actors (`inspect_world info`).
- M1: `--no-viewer --duration 1800 --auto-clear 30` ends with `collisions 0` in the summary for each robotaxi and trips > 0.
- S1: pressing `T` ten times (with `C` in between) makes the selected robotaxi stop before the cones and open a `LANE_BLOCKED` request every time.
- Sites: within a few minutes every site shows up in the SCENARIOS panel as ACTIVE when a robotaxi heads there; the incident appears out of camera range, and is gone once the robotaxis have passed.
- M4: in S1, `BYPASS_LEFT` passes the zone without touching a cone and `BYPASS_RIGHT` is grey with a reason; `PULL_OVER` then `RESUME` returns to the trip; `ABORT` during a bypass holds the car and `RESUME` finishes it.
- S2: the robotaxi stops ~10 m behind the van, opens `STOPPED_VEHICLE` after 5 s; `BYPASS_LEFT` waits for oncoming traffic (two-lane roads) and passes.
- S3: the robotaxi stops before the junction with `ROAD_CLOSED`; only turns that lead to the destination are green; the car turns and does not come back to the closed road.
- S4: every few minutes a random robotaxi with a passenger asks for help while driving; `PULL_OVER` stops at the curb, `RESUME` continues.
- S5: `BYPASS_RIGHT` is green, `BYPASS_LEFT` is grey.
- Viewer: front view with three mirrors; `M` hides them; `V` cycles the views.

## Driving details worth knowing

- **Junctions**: the car looks across the next junction (25 m past its exit), so a closure right behind a long junction stops it at the junction entry, never inside.
- **Lane changes**: CARLA's BasicAgent changes lanes without looking. The robotaxi waits (or slows) until it can merge behind a car in the target lane; this matters most right after junctions, where the route planner puts many lane changes.
- **Following**: besides BasicAgent's lane check, any vehicle in the route corridor is followed at 4 m, also across road boundaries and inside junctions (e.g. a robotaxi dropping off in its lane).
- **S3 turns**: a turn that leads back into the closed road is never offered. If the destination can only be reached through the closure, the turn is still offered and the trip ends shortly after the junction (drop-off moved).

## Tuning the bypass

The defaults (`lateral_margin_m: 0.7`, `tracking_lag_m: 4.0`, `rear_margin_m: 1.5`,
`obstacle_standoff_m: 6.0`) were checked on a kinematic test model with a
junction and an oncoming lane: no contacts, about 0.57 m minimum clearance to
props. If a robotaxi still touches cones in CARLA, raise `lateral_margin_m` or
`tracking_lag_m` in `config/fleet.yaml` (or the standoff, so it stops further
back and has room to steer). Turn on `D` -> full to see the planned path.

## Troubleshooting

- **Letter keys do nothing**: switch the input method to English; or use the buttons.

- **Scenario stays SEARCHING**: the viewer shows the reason (route too short, side-lane requirement, someone in front). It retries every second as the car drives on; relax `placement` in the YAML if a map has few long straight roads.
- **Sign faces the wrong way**: change `yaw_offset_deg` in `S1_construction.yaml`.
- **A site never becomes ACTIVE**: no robotaxi route runs through it. Raise `mission.via_site_probability`, check the site in the log at start-up (`site A road 10 lane -2 s 60.0`; a warning means the coordinates are not on a driving lane), or turn on `D` -> full to see where it is.
- **`F3` (S3 injected by hand) never starts**: it needs a junction 40-160 m ahead that the route crosses straight and that offers a turn. Press it again later, or relax `route_through` / `require_turn` in `S3_road_closed.yaml`.
- **Only `PULL_OVER` is green**: e.g. the trip ends right behind the incident. Pull over (drop-off there) and `RESUME`.
- **A blueprint is missing on your map** (e.g. the van): the `fallback_blueprints` list is tried in order; `inspect_world props <name>` lists what exists.
- **Server stuck after a crash** (world frozen in synchronous mode): restart the demo, or run `python -m ra_demo --duration 1`, which restores the settings on exit.

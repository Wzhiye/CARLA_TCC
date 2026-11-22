"""Interactive point picker for scenario authoring.

Fly the spectator in the CARLA server window (hold the right mouse button and
use W/A/S/D/Q/E), then type a command here:

    <Enter>   capture: print the nearest driving-lane waypoint and save it
    s         label spawn points within 80 m with their index
    l         label lanes within 40 m with road/lane ids
    q         quit

Captured points are appended to ``tools/picked_points.yaml`` in a format that
can be pasted into scenario and fleet files.

The tool never ticks the world. If the demo is running it ticks for us;
otherwise the server is in asynchronous mode and draws on its own.
"""

import argparse
import math
import os
import time

import carla
import yaml

from ra_demo.core import compat

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL_TIME = 30.0


def waypoint_record(world_map, location, spawn_points):
    waypoint = world_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    nearest_index, nearest_distance = None, float("inf")
    for index, point in enumerate(spawn_points):
        distance = point.location.distance(waypoint.transform.location)
        if distance < nearest_distance:
            nearest_index, nearest_distance = index, distance
    tf = waypoint.transform
    return waypoint, {
        "location": [round(tf.location.x, 2), round(tf.location.y, 2), round(tf.location.z, 2)],
        "yaw": round(tf.rotation.yaw, 1),
        "road_id": waypoint.road_id,
        "section_id": waypoint.section_id,
        "lane_id": waypoint.lane_id,
        "s": round(waypoint.s, 2),
        "lane_width": round(waypoint.lane_width, 2),
        "is_junction": waypoint.is_junction,
        "left_lane": compat.describe_side_lane(waypoint, "left"),
        "right_lane": compat.describe_side_lane(waypoint, "right"),
        "lane_change": str(waypoint.lane_change),
        "nearest_spawn_point": nearest_index,
        "nearest_spawn_distance_m": round(nearest_distance, 1),
    }


def capture(world, world_map, spawn_points, out_path):
    spectator = world.get_spectator().get_transform()
    waypoint, record = waypoint_record(world_map, spectator.location, spawn_points)
    record["captured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    record["map"] = compat.short_map_name(world_map.name)
    print(yaml.safe_dump(record, sort_keys=False, allow_unicode=True))
    location = waypoint.transform.location + carla.Location(z=0.5)
    world.debug.draw_point(location, size=0.2, color=carla.Color(255, 0, 255), life_time=LABEL_TIME)
    world.debug.draw_string(location + carla.Location(z=1.0),
                            "r{} l{} s{:.0f}".format(record["road_id"], record["lane_id"], record["s"]),
                            color=carla.Color(255, 0, 255), life_time=LABEL_TIME)
    existing = []
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as handle:
            existing = yaml.safe_load(handle) or []
    existing.append(record)
    with open(out_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(existing, handle, sort_keys=False, allow_unicode=True)
    print("saved to {} ({} points)".format(out_path, len(existing)))


def label_spawn_points(world, spawn_points, radius=80.0):
    center = world.get_spectator().get_transform().location
    shown = 0
    for index, point in enumerate(spawn_points):
        if point.location.distance(center) > radius:
            continue
        world.debug.draw_string(point.location + carla.Location(z=1.5), str(index),
                                color=carla.Color(0, 255, 0), life_time=LABEL_TIME)
        forward = point.get_forward_vector()
        start = point.location + carla.Location(z=0.5)
        end = carla.Location(start.x + forward.x * 3.0, start.y + forward.y * 3.0, start.z)
        world.debug.draw_arrow(start, end, thickness=0.1, arrow_size=0.3,
                               color=carla.Color(0, 255, 0), life_time=LABEL_TIME)
        shown += 1
    print("labelled {} spawn points within {:.0f} m".format(shown, radius))


def label_lanes(world, world_map, radius=40.0):
    center = world.get_spectator().get_transform().location
    for waypoint in world_map.generate_waypoints(6.0):
        location = waypoint.transform.location
        if math.hypot(location.x - center.x, location.y - center.y) > radius:
            continue
        world.debug.draw_string(location + carla.Location(z=0.5),
                                "r{}/l{}".format(waypoint.road_id, waypoint.lane_id),
                                color=carla.Color(255, 200, 0), life_time=LABEL_TIME)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--out", default=os.path.join(HERE, "picked_points.yaml"))
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    world_map = world.get_map()
    spawn_points = world_map.get_spawn_points()
    print("Connected to {} ({} spawn points)".format(compat.short_map_name(world_map.name),
                                                     len(spawn_points)))
    if world.get_settings().synchronous_mode:
        print("Note: the world is in synchronous mode. Labels appear only while the demo is ticking.")

    while True:
        command = input("[Enter]=capture  s=spawn points  l=lanes  q=quit > ").strip().lower()
        if command == "q":
            break
        if command == "s":
            label_spawn_points(world, spawn_points)
        elif command == "l":
            label_lanes(world, world_map)
        else:
            capture(world, world_map, spawn_points, args.out)


if __name__ == "__main__":
    main()

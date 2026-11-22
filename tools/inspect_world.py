"""Print facts about the running CARLA world (read-only).

    python -m ra_demo.tools.inspect_world info
    python -m ra_demo.tools.inspect_world maps
    python -m ra_demo.tools.inspect_world props            # all static.prop.* ids
    python -m ra_demo.tools.inspect_world props cone       # filter by substring
    python -m ra_demo.tools.inspect_world vehicles
"""

import argparse

import carla

from ra_demo.core import compat


def cmd_info(client, world, _):
    settings = world.get_settings()
    world_map = world.get_map()
    print("client version :", compat.client_version(client))
    print("server version :", compat.server_version(client))
    print("map            :", compat.short_map_name(world_map.name))
    print("spawn points   :", len(world_map.get_spawn_points()))
    print("synchronous    :", settings.synchronous_mode, " fixed_delta:", settings.fixed_delta_seconds)
    print("no rendering   :", settings.no_rendering_mode)
    actors = world.get_actors()
    print("vehicles       :", len(actors.filter("vehicle.*")))
    print("walkers        :", len(actors.filter("walker.pedestrian.*")))
    print("static props   :", len(actors.filter("static.prop.*")))


def cmd_maps(client, _world, _):
    for name in sorted(client.get_available_maps()):
        print(name)


def cmd_props(_client, world, args):
    needle = (args.filter or "").lower()
    for blueprint in sorted(world.get_blueprint_library().filter("static.prop.*"), key=lambda b: b.id):
        if needle in blueprint.id.lower():
            print(blueprint.id)


def cmd_vehicles(_client, world, args):
    needle = (args.filter or "").lower()
    for blueprint in sorted(world.get_blueprint_library().filter("vehicle.*"), key=lambda b: b.id):
        if needle not in blueprint.id.lower():
            continue
        base_type = blueprint.get_attribute("base_type").as_str() if blueprint.has_attribute("base_type") else "?"
        colors = "color" if blueprint.has_attribute("color") else ""
        print("{:<45} {:<12} {}".format(blueprint.id, base_type, colors))


COMMANDS = {"info": cmd_info, "maps": cmd_maps, "props": cmd_props, "vehicles": cmd_vehicles}


def main():
    parser = argparse.ArgumentParser(description="Inspect the running CARLA world")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("filter", nargs="?", help="substring filter for props/vehicles")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    COMMANDS[args.command](client, client.get_world(), args)


if __name__ == "__main__":
    main()

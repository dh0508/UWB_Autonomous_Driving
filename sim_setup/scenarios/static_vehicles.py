#!/usr/bin/env python

# Copyright (c) 2025 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn 60 vehicles driving with normal autopilot: 30 'collector' vehicles
that just drive normally, and 30 'stopper' vehicles that alternate between
driving and stopping at random for a while. Every traffic light in the map
is frozen to green for the whole run."""

import carla

from carla import VehicleLightState as vls
from carla.command import SpawnActor, SetAutopilot, FutureActor, DestroyActor

import argparse
import logging
from numpy import random
import time


def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--num-collector-vehicles',
        metavar='N',
        default=15,
        type=int,
        help='Number of vehicles that just drive normally (default: 30)')
    argparser.add_argument(
        '--num-stopper-vehicles',
        metavar='N',
        default=15,
        type=int,
        help='Number of vehicles that drive but stop at random intervals (default: 30)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=0,
        type=int,
        help='Number of walkers (default: 0)')
    argparser.add_argument(
        '--stop-min-interval',
        metavar='S',
        default=8.0,
        type=float,
        help='Minimum seconds a stopper vehicle drives before stopping (default: 8.0)')
    argparser.add_argument(
        '--stop-max-interval',
        metavar='S',
        default=20.0,
        type=float,
        help='Maximum seconds a stopper vehicle drives before stopping (default: 20.0)')
    argparser.add_argument(
        '--stop-min-duration',
        metavar='S',
        default=5.0,
        type=float,
        help='Minimum seconds a stopper vehicle stays stopped, like a red light (default: 5.0)')
    argparser.add_argument(
        '--stop-max-duration',
        metavar='S',
        default=5.0,
        type=float,
        help='Maximum seconds a stopper vehicle stays stopped, like a red light (default: 5.0)')
    argparser.add_argument(
        '--normal-speed-diff',
        metavar='P',
        default=30.0,
        type=float,
        help='Traffic Manager percentage_speed_difference used while a stopper is driving normally (default: 30.0)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='Filter vehicle model (default: "vehicle.*")')
    argparser.add_argument(
        '--generationv',
        metavar='G',
        default='All',
        help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='Filter pedestrian type (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--generationw',
        metavar='G',
        default='2',
        help='restrict to certain pedestrian generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='Port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--asynch',
        action='store_true',
        help='Activate asynchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Activate hybrid mode for Traffic Manager')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        type=int,
        help='Set random device seed and deterministic mode for Traffic Manager')
    argparser.add_argument(
        '--seedw',
        metavar='S',
        default=0,
        type=int,
        help='Set the seed for pedestrians module')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enable automatic car light management')
    argparser.add_argument(
        '--hero',
        action='store_true',
        default=False,
        help='Set one of the collector vehicles as hero')
    argparser.add_argument(
        '--respawn',
        action='store_true',
        default=False,
        help='Automatically respawn dormant vehicles (only in large maps)')
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        default=False,
        help='Activate no rendering mode')

    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    vehicle_roles = {}
    walkers_list = []
    all_id = []
    frozen_lights = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    random.seed(args.seed if args.seed is not None else int(time.time()))

    original_world_settings = None
    try:
        world = client.get_world()

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        if args.respawn:
            traffic_manager.set_respawn_dormant_vehicles(True)
        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(70.0)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)

        original_world_settings = world.get_settings()
        print("current_world_settings {}".format(original_world_settings))
        settings = original_world_settings
        if not args.asynch:
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
        else:
            print("You are currently in asynchronous mode. If this is a traffic simulation, \
            you could experience some issues. If it's not working correctly, switch to synchronous \
            mode by using traffic_manager.set_synchronous_mode(True)")

        if args.no_rendering:
            settings.no_rendering_mode = True
        print("apply_world_settings {}".format(settings))
        world.apply_settings(settings)
        print("settings applied")

        # -----------------------------------------
        # Freeze every traffic light to green
        # -----------------------------------------
        for light in world.get_actors():
            if not isinstance(light, carla.TrafficLight):
                continue
            light.freeze(False)  # unfreeze first: state changes on a frozen light can be ignored
            light.set_state(carla.TrafficLightState.Green)
            light.set_green_time(1e6)
            light.set_red_time(0.0)
            light.set_yellow_time(0.0)
            light.freeze(True)
            frozen_lights.append(light)
        # push the state change through before anything else spawns/ticks
        if args.asynch:
            world.wait_for_tick()
        else:
            world.tick()
        print('froze %d traffic lights to green' % len(frozen_lights))

        blueprints = get_actor_blueprints(world, args.filterv, args.generationv)
        if not blueprints:
            raise ValueError("Couldn't find any vehicles with the specified filters")
        blueprintsWalkers = get_actor_blueprints(world, args.filterw, args.generationw)
        if args.number_of_walkers > 0 and not blueprintsWalkers:
            raise ValueError("Couldn't find any walkers with the specified filters")

        if args.safe:
            blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car']

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        number_of_vehicles = args.num_collector_vehicles + args.num_stopper_vehicles
        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, number_of_vehicles, number_of_spawn_points)
            # keep the 50/50 collector/stopper split while fitting the map
            number_of_vehicles = number_of_spawn_points
            args.num_collector_vehicles = number_of_vehicles // 2
            args.num_stopper_vehicles = number_of_vehicles - args.num_collector_vehicles

        # ----------------------------------------------------
        # Spawn vehicles with autopilot: collectors then stoppers
        # ----------------------------------------------------
        batch = []
        batch_roles = []
        hero = args.hero
        for n, transform in enumerate(spawn_points):
            if n >= number_of_vehicles:
                break
            role = 'collector' if n < args.num_collector_vehicles else 'stopper'

            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if hero:
                blueprint.set_attribute('role_name', 'hero')
                hero = False
            else:
                blueprint.set_attribute('role_name', role)

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))
            batch_roles.append(role)

        stopper_ids = []
        for i, response in enumerate(client.apply_batch_sync(batch, do_tick=True)):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)
                vehicle_roles[response.actor_id] = batch_roles[i]
                if batch_roles[i] == 'stopper':
                    stopper_ids.append(response.actor_id)

        # Set automatic vehicle lights update if specified
        if args.car_lights_on:
            all_vehicle_actors = world.get_actors(vehicles_list)
            for actor in all_vehicle_actors:
                traffic_manager.update_vehicle_lights(actor, True)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        if args.seedw:
            world.set_pedestrians_seed(args.seedw)
            random.seed(args.seedw)
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            probability = random.randint(0,100 + 1);
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            if walker_bp.has_attribute('can_use_wheelchair') and probability < 11:
                walker_bp.set_attribute('use_wheelchair', 'true')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, do_tick=True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, do_tick=True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if args.asynch:
            world.wait_for_tick()
        else:
            world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        # -----------------------------------------------------
        # Stop/go scheduler for the 'stopper' vehicles: each one keeps
        # driving on autopilot the whole time. To stop, we don't touch
        # its controls directly (a manual full-brake/hand-brake snap while
        # the Traffic Manager is still steering it is what caused the
        # vehicles to launch/flip) - instead we tell the Traffic Manager to
        # target 0 speed for that vehicle, so it brakes down and holds
        # exactly the way it already does for a red light. Resuming just
        # hands the target speed back to normal.
        # -----------------------------------------------------
        stopper_actors = {aid: world.get_actor(aid) for aid in stopper_ids}
        now = world.get_snapshot().timestamp.elapsed_seconds
        stopper_state = {}
        for aid in stopper_ids:
            stopper_state[aid] = {
                'stopped': False,
                # stagger everyone's first stop so they don't all halt together
                'next_toggle': now + random.uniform(args.stop_min_interval, args.stop_max_interval),
            }

        def toggle_stopper(actor, state):
            if state['stopped']:
                # resume: back to the normal cruising speed target
                traffic_manager.vehicle_percentage_speed_difference(actor, args.normal_speed_diff)
                state['stopped'] = False
                state['next_toggle'] = now + random.uniform(args.stop_min_interval, args.stop_max_interval)
            else:
                # stop: 100% below speed limit == target speed 0, TM brakes
                # it to a smooth stop and holds it there like a red light
                traffic_manager.vehicle_percentage_speed_difference(actor, 100.0)
                state['stopped'] = True
                state['next_toggle'] = now + random.uniform(args.stop_min_duration, args.stop_max_duration)

        print('spawned %d vehicles (%d collector, %d stopper) and %d walkers, press Ctrl+C to exit.' % (
            len(vehicles_list), args.num_collector_vehicles, len(stopper_ids), len(walkers_list)))

        # Example of how to use Traffic Manager parameters
        traffic_manager.global_percentage_speed_difference(args.normal_speed_diff)

        while True:
            if not args.asynch:
                world.tick()
            else:
                world.wait_for_tick()

            now = world.get_snapshot().timestamp.elapsed_seconds
            for aid in stopper_ids:
                actor = stopper_actors.get(aid)
                if actor is None or not actor.is_alive:
                    continue
                state = stopper_state[aid]
                if now >= state['next_toggle']:
                    toggle_stopper(actor, state)

    finally:

        if frozen_lights:
            print('\nunfreezing %d traffic lights' % len(frozen_lights))
            for light in frozen_lights:
                light.freeze(False)

        if not args.asynch:
            if original_world_settings:
                settings= original_world_settings
            else:
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.no_rendering_mode = False
                settings.fixed_delta_seconds = None
            print("restore world_settings {}".format(settings))
            world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([DestroyActor(x) for x in all_id])

        time.sleep(0.5)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')

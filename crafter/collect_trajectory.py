import os
from os.path import join as pjoin
import uuid
import json
import argparse
import copy

import numpy as np
try:
    import pygame
except ImportError:
    print('Please install the pygame package to use the GUI.')
    raise
from PIL import Image

import crafter


def main():
    boolean = lambda x: bool(['False', 'True'].index(x))
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--area', nargs=2, type=int, default=(64, 64))
    parser.add_argument('--view', type=int, nargs=2, default=(9, 9))
    parser.add_argument('--length', type=int, default=None)
    parser.add_argument('--health', type=int, default=9)
    parser.add_argument('--window', type=int, nargs=2, default=(600, 600))
    parser.add_argument('--size', type=int, nargs=2, default=(0, 0))
    parser.add_argument('--record', type=str, default=None)
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--wait', type=boolean, default=False)
    parser.add_argument('--death', type=str, default='quit', choices=[
            'continue', 'reset', 'quit'])
    args = parser.parse_args()

    keymap = {
            pygame.K_a: 'move_left',
            pygame.K_d: 'move_right',
            pygame.K_w: 'move_up',
            pygame.K_s: 'move_down',
            pygame.K_SPACE: 'do',
            pygame.K_TAB: 'sleep',

            pygame.K_r: 'place_stone',
            pygame.K_t: 'place_table',
            pygame.K_f: 'place_furnace',
            pygame.K_p: 'place_plant',

            pygame.K_1: 'make_wood_pickaxe',
            pygame.K_2: 'make_stone_pickaxe',
            pygame.K_3: 'make_iron_pickaxe',
            pygame.K_4: 'make_wood_sword',
            pygame.K_5: 'make_stone_sword',
            pygame.K_6: 'make_iron_sword',
    }
    print('Actions:')
    for key, action in keymap.items():
        print(f'    {pygame.key.name(key)}: {action}')

    crafter.constants.items['health']['max'] = args.health
    crafter.constants.items['health']['initial'] = args.health

    size = list(args.size)
    size[0] = size[0] or args.window[0]
    size[1] = size[1] or args.window[1]

    env = crafter.Env(
            area=args.area, view=args.view, length=args.length, seed=args.seed)
    env = crafter.Recorder(env, args.record)
    init_obs = env.reset()

    # create the data folder
    _seed = env._seed
    os.makedirs("data", exist_ok=True)
    # random uuid for this trajectory
    _uuid = str(uuid.uuid4())
    output_file = pjoin("data", f"trajectory_{_uuid}_seed_{_seed}.jsonl")
    collected_trajectory = []
    action_counter = {}

    collected_trajectory.append({"s_t": [copy.copy(init_obs[1]), copy.copy(init_obs[2]), copy.deepcopy(env._player._internal_counters)]})
    achievements = set()
    duration = 0
    return_ = 0
    was_done = False
    total_reward = 0
    print('Diamonds exist:', env._world.count('diamond'))

    pygame.init()
    screen = pygame.display.set_mode(args.window)
    clock = pygame.time.Clock()
    running = True
    while running:

        # Rendering.
        image = env.render(size)[0]
        if size != args.window:
            image = Image.fromarray(image)
            image = image.resize(args.window, resample=Image.NEAREST)
            image = np.array(image)
        surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(args.fps)

        # Keyboard input.
        action = None
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in keymap.keys():
                action = keymap[event.key]
        if action is None:
            pressed = pygame.key.get_pressed()
            for key, action in keymap.items():
                if pressed[key]:
                    break
            else:
                if args.wait and not env._player.sleeping:
                    continue
                else:
                    action = 'noop'

        # Environment step.
        obs, reward, done, _ = env.step(env.action_names.index(action))
        if action != "noop":
            total_reward += reward
            collected_trajectory[-1]["r_t"] = reward
            collected_trajectory[-1]["cumulative_r_t"] = total_reward
            collected_trajectory[-1]["a_t"] = action
            collected_trajectory[-1]["s_t+1"] = [copy.copy(obs[1]), copy.copy(obs[2]), copy.deepcopy(env._player._internal_counters)]
            collected_trajectory[-1]["done"] = done

            action_success = False
            if collected_trajectory[-1]["s_t+1"][0] != collected_trajectory[-1]["s_t"][0]:
                action_success = True
            if collected_trajectory[-1]["s_t+1"][1] != collected_trajectory[-1]["s_t"][1]:
                if collected_trajectory[-1]["s_t+1"][1]["health"] >= collected_trajectory[-1]["s_t"][1]["health"] or \
                    collected_trajectory[-1]["s_t+1"][1]["food"] >= collected_trajectory[-1]["s_t"][1]["food"] or \
                    collected_trajectory[-1]["s_t+1"][1]["drink"] >= collected_trajectory[-1]["s_t"][1]["drink"] or \
                    collected_trajectory[-1]["s_t+1"][1]["energy"] >= collected_trajectory[-1]["s_t"][1]["energy"]:
                    action_success = True
            if action not in action_counter:
                action_counter[action] = {"success": 0, "fail": 0}
            action_counter[action]["success" if action_success else "fail"] += 1
            collected_trajectory[-1]["action_counter"] = copy.deepcopy(action_counter)
            collected_trajectory.append({"s_t": [copy.copy(obs[1]), copy.copy(obs[2]), copy.deepcopy(env._player._internal_counters)]})
            # print("====================================")
            # print(obs[1])    # text matrix
            # print(obs[2])    # inventory
        duration += 1

        # Achievements.
        unlocked = {
                name for name, count in env._player.achievements.items()
                if count > 0 and name not in achievements}
        for name in unlocked:
            achievements |= unlocked
            total = len(env._player.achievements.keys())
            print(f'Achievement ({len(achievements)}/{total}): {name}')
        if env._step > 0 and env._step % 100 == 0:
            print(f'Time step: {env._step}')
        if reward:
            print(f'Reward: {reward}')
            return_ += reward

        # Episode end.
        if done and not was_done:
            was_done = True
            print('Episode done!')
            print('Duration:', duration)
            print('Return:', return_)
            if args.death == 'quit':
                running = False
            if args.death == 'reset':
                print('\nStarting a new episode.')
                env.reset()
                achievements = set()
                was_done = False
                duration = 0
                return_ = 0
            if args.death == 'continue':
                pass

    pygame.quit()
    collected_trajectory[-1]["a_t"] = "none"
    collected_trajectory[-1]["s_t+1"] = ["none", "none"]
    # write collected trajectory into json file
    with open(output_file, "w") as f:
        f.write(json.dumps({"seed": _seed, "log": collected_trajectory}, indent=4))
    print(f"Trajectory saved to {output_file}")


if __name__ == '__main__':
    main()

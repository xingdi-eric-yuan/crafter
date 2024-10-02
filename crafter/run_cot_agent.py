import os
import datetime
import copy
import json
import crafter
from llm_api import LLM
MAX_TRY = 20


class Actor:

    _available_actions = ['idle', 'move_west', 'move_east', 'move_north', 'move_south', 'sleep', \
                         'place_stone', 'place_table', 'place_furnace', 'place_plant', 'make_wood_pickaxe', \
                         'make_stone_pickaxe', 'make_iron_pickaxe', 'make_wood_sword', 'make_stone_sword', \
                         'make_iron_sword', 'collect_coal', 'collect_diamond', 'collect_drink', \
                         'collect_iron', 'collect_sapling', 'collect_stone', 'collect_wood', \
                         'defeat_skeleton', 'defeat_zombie', 'eat_cow', 'eat_plant']
    action_mapping = {
        'idle': 0,
        'move_west': 1,
        'move_east': 2,
        'move_north': 3,
        'move_south': 4,
        'do': 5,
        'sleep': 6,
        'place_stone': 7,
        'place_table': 8,
        'place_furnace': 9,
        'place_plant': 10,
        'make_wood_pickaxe': 11,
        'make_stone_pickaxe': 12,
        'make_iron_pickaxe': 13,
        'make_wood_sword': 14,
        'make_stone_sword': 15,
        'make_iron_sword': 16,
        }

    def __init__(self, 
                 initial_size=(600, 600), map_size=64, 
                 seed=0):

        self._map_size = map_size
        self._initial_size = initial_size
        self._seed = seed
        self.llm = LLM("gpt-4o")
        
        self._env = crafter.Env(seed=self._seed)
        self._env = crafter.Recorder(
            self._env, None,
            save_stats=False,
            save_episode=False,
            save_video=False,
        )
        
    def map_action(self, text_action):
        if 'face' in text_action:
            text_action = text_action.replace('face', 'move')
        for action, number in self.action_mapping.items():
            if action in text_action:
                return action, number
        return 'do(mine, collect, attack)', 5

    def reset(self):
        self._time_step = 0
        self._total_reward = 0.0
        self.transition_trajectory = []
        self._env.reset()
        obs = self._env.render()
        self.transition_trajectory.append({"s_t": [copy.copy(obs[1]), copy.copy(obs[2])]})
        return obs

    def step(self, text_action):
        _, mapped_action = self.map_action(text_action)   
        obs, reward, done, _ = self._env.step(mapped_action)
        self._total_reward += reward
        self.transition_trajectory[-1]["a_t"] = text_action
        self.transition_trajectory[-1]["r_t"] = reward
        self.transition_trajectory[-1]["s_t+1"] = [copy.copy(obs[1]), copy.copy(obs[2])]
        self.transition_trajectory[-1]["done"] = done
        self.transition_trajectory.append({"s_t": [copy.copy(obs[1]), copy.copy(obs[2])]})
        return obs, reward, done

    def run(self):

        done = False
        obs = self.reset()
        print('============================')
        print("game started...")
        while not done:
            _, text_matrix, inventory = obs
            llm_response = self.act()
            text_action = llm_response["chosen_action"]
            obs, _, done = self.step(text_action)

            self._time_step += 1
            print("step:", self._time_step)
            print('total reward:', self._total_reward)
            print('state: \n', self.observation_layout(text_matrix))
            print('inventory: \n', inventory)
            print('action', text_action)
            print('============================')
        print("game over...")

    def act(self):
        if self._env._player.sleeping:
            return {"chosen_action": "idle"}
        return self.select_action()

    def observation_layout(self, text_matrix):
        # text_matrix is a list of list
        output = []
        for i in range(len(text_matrix)):
            output.append(', '.join(text_matrix[i]))
        return ';\n'.join(output)

    def select_action(self):
        messages = []
        messages.append({"role": "system", "content" : "You are a helpful assistant. You are playing a 2-d grid-based game. Your task is to select the best action to complete the final goal based on the game state at every step. The game state has been translated into a grid of text for your convenience. Please provide your answer in JSON format."})
        

        action_format = """{
            'top_3_actions_proposal': {'action_name_1': {'object_1_name': {'location': object 1's location, 'dynamic': object 1's dynamic}, ......}
                                        'action_name_2': ......
                                        'action_name_3': ......},
            'top_3_actions_consequences': {'action_name_1': 'consequences', 'action_name_2': 'consequences', 'action_name_3': 'consequences'},
            'chosen_action': 'action_name',
            'justification': 'justification'
        }"""
        prompt = f"""
        Given the following details:
        - Final game goal: survive and collect the diamond
        - Current observation: {json.dumps(self.transition_trajectory[-1]["s_t"][0])}
        - Current status: {json.dumps(self.transition_trajectory[-1]["s_t"][1])}
        - Previous actions: {json.dumps([self.transition_trajectory[i]["a_t"] for i in range(len(self.transition_trajectory) - 1)][-3:])}

        You task is to:
        - propose the top 3 actions to execute at next step based on the current observation and status.
        - provide the rationale and detailed consequences of executing each action you propose.
        - based on the consequence, select the best action to execute next.
        - provide the justification for your choice.

        Note: You can only select actions from the available actions: {json.dumps(self._available_actions)}.
        Note: Avoid unnecessary crafting and placement if the items are within reachable distance.
        Note: You should craft as many different tools as soon as possible to increase the chance of reaching the final goal.
        
        Please format your response in the following format: {action_format}
        """
        messages.append({"role": "user", "content": prompt})

        retried_times = 0 
        while True:
            if retried_times >= MAX_TRY:
                return {"chosen_action": "idle"}
            response = self.llm(messages, json_format=True)
            try:
                response_json = json.loads(response)
                action = response_json['chosen_action']
                assert action in self._available_actions
                break
            except:
                print("error in json.loads, retrying...")
                retried_times += 1
        return response_json


if __name__ == '__main__':
    actor = Actor()
    actor.run()

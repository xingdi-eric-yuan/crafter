import os
import datetime
import copy
import uuid
import json
import crafter
from os.path import join as pjoin
from llm_api import LLM
MAX_TRY = 20
HISTORY_SIZE = 5


class Actor:

    _available_actions = {
        'idle': 0,
        'move_left': 1,
        'move_right': 2,
        'move_up': 3,
        'move_down': 4,
        'do (mine, collect, attack)': 5,
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
                 step_budget=1000, agent_type="plain",
                 seed=0):

        self._map_size = map_size
        self._initial_size = initial_size
        self._seed = seed
        self.agent_type = agent_type
        self.step_budget = step_budget
        self.llm = LLM("gpt-4o")
        
        self._env = crafter.Env(seed=self._seed)
        self._env = crafter.Recorder(
            self._env, None,
            save_stats=False,
            save_episode=False,
            save_video=False,
        )

    def save_log(self):
        with open(self.output_file_path, "w") as f:
            f.write(json.dumps({"seed": self._seed, "log": self.log}, indent=4))

    def map_action(self, text_action):
        if text_action in self._available_actions:
            return self._available_actions[text_action]
        else:
            return 5

    def reset(self):
        self._time_step = 0
        self._total_reward = 0.0
        self.transition_trajectory = []
        self.action_counter = {}
        for __action in self._available_actions.keys():
            self.action_counter[__action] = {"success": 0, "fail": 0}
        self._env.reset()
        obs = self._env.render()
        self.transition_trajectory.append({"s_t": [copy.deepcopy(obs[1]), copy.deepcopy(obs[2]), copy.deepcopy(self._env._player._internal_counters)]})

        # log
        os.makedirs("log", exist_ok=True)
        # random uuid for this run
        _uuid = str(uuid.uuid4())
        self.output_file_path = pjoin("log", f"{_uuid}_seed_{self._seed}.jsonl")
        self.log = []

        return obs

    def step(self, llm_response):
        text_action = llm_response["chosen_action"]
        action_id = self.map_action(text_action)   
        obs, reward, done, _ = self._env.step(action_id)
        self._total_reward += reward
        self.transition_trajectory[-1]["a_t"] = copy.deepcopy(llm_response)
        self.transition_trajectory[-1]["r_t"] = reward
        self.transition_trajectory[-1]["cumulative_r_t"] = self._total_reward
        self.transition_trajectory[-1]["s_t+1"] = [copy.deepcopy(obs[1]), copy.deepcopy(obs[2]), copy.deepcopy(self._env._player._internal_counters)]
        self.transition_trajectory[-1]["done"] = done
        action_success = False
        if self.transition_trajectory[-1]["s_t+1"][0] != self.transition_trajectory[-1]["s_t"][0]:
            action_success = True
        if self.transition_trajectory[-1]["s_t+1"][1] != self.transition_trajectory[-1]["s_t"][1]:
            if self.transition_trajectory[-1]["s_t+1"][1]["health"] != self.transition_trajectory[-1]["s_t"][1]["health"] or \
                self.transition_trajectory[-1]["s_t+1"][1]["food"] != self.transition_trajectory[-1]["s_t"][1]["food"] or \
                self.transition_trajectory[-1]["s_t+1"][1]["drink"] != self.transition_trajectory[-1]["s_t"][1]["drink"] or \
                self.transition_trajectory[-1]["s_t+1"][1]["energy"] != self.transition_trajectory[-1]["s_t"][1]["energy"]:
                if self._env.player.internal_status_change is False:
                    action_success = True
        self.action_counter[text_action]["success" if action_success else "fail"] += 1
        self.transition_trajectory[-1]["action_counter"] = copy.deepcopy(self.action_counter)
        self.transition_trajectory.append({"s_t": [copy.deepcopy(obs[1]), copy.deepcopy(obs[2]), copy.deepcopy(self._env._player._internal_counters)]})
        return obs, reward, done

    def run(self):

        done = False
        obs = self.reset()
        print("game started...")
        print('============================')
        print('- state: \n', self.observation_layout(copy.deepcopy(obs[1])))
        print('- inventory: \n', copy.deepcopy(obs[2]))
        while not done:
            if self._time_step >= self.step_budget:
                break
            llm_response = self.act()
            obs, _, done = self.step(llm_response)
            print('============================')
            print("- step:", self._time_step)
            print('- action:', llm_response["chosen_action"])
            print('- state: \n', self.observation_layout(copy.deepcopy(obs[1])))
            print('- inventory: \n', copy.deepcopy(obs[2]))
            print('- total reward:', self._total_reward)
            self._time_step += 1
            self.log.append(copy.deepcopy(self.transition_trajectory[-2]))
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
        messages.append({"role": "system", "content" : "You are a helpful assistant. You are playing Crafter, a 2-d grid-world game. Your final goal is to survive and collect the diamond, but to achieve that, at every step, your task is to select the best action towards the final goal based on the current game state. The game state has been translated into a grid of text for your convenience. Please provide your answer in JSON format."})

        action_format_cot = {
            "top_3_actions_proposal": {"action_1": "the action you propose", 
                                       "action_2": "the action you propose", 
                                       "action_3": "the action you propose"},
            "top_3_actions_nationale": {"action_1": "why you propose this action", 
                                        "action_2": "why you propose this action", 
                                        "action_3": "why you propose this action"},
            'top_3_actions_consequences': {"action_1": "what will happen if you execute this action", 
                                           "action_2": "what will happen if you execute this action",
                                           "action_3": "what will happen if you execute this action"},
            "chosen_action": "the action you choose to execute",
            "justification": "the reason why you choose this action from the top 3 actions"
        }

        action_format_plain = {
            "chosen_action": "the action you choose to execute",
        }

        game_info = f"""
The game Crafter is a 2-d grid-world game where you play as a character who can move around, mine resources, craft tools, and place objects. The game world is a grid of size {self._map_size}x{self._map_size}, but the player can only observe their surroundings of size 9x7. The game state is represented by a grid of text where each cell contains information about the material at that location. The game state also provides the player's inventory, which includes information about the player's health, food, drink, energy level, as well as the tools and resources they have collected. The player's goal is to survive and collect the diamond, which is hidden somewhere in the game world. When the food, drink, or energy level reaches zero, the player will lose health. The player needs to find ways to replenish these resources to stay alive. The player can also craft tools to help them mine resources faster and survive longer. The game has a tech-tree where the player needs to craft lower-tier tools before they can craft higher-tier tools.
"""

        prompt_plain = f"""
Given the following information describing the current game state:
- Current observation: 
    {json.dumps(self.transition_trajectory[-1]["s_t"][0])}
- Current status: 
    {json.dumps(self.transition_trajectory[-1]["s_t"][1])}
- Previous actions: 
    {json.dumps([self.transition_trajectory[i]["a_t"] for i in range(len(self.transition_trajectory) - 1)][-HISTORY_SIZE:])}
- Rewards received so far: 
    {self._total_reward}

You task is to propose the best actions to execute at next step based on the current observation and status.

Note: You can only select actions from the available actions: {json.dumps(self._available_actions)}.
Note: Avoid unnecessary crafting and placement if the items are within reachable distance.
Note: You should craft as many different tools as soon as possible to increase the chance of reaching the final goal.

Please format your response in the following format: \n{json.dumps(action_format_plain, indent=4)}
"""

        prompt_cot = f"""
Given the following information describing the current game state:
- Current observation: 
    {json.dumps(self.transition_trajectory[-1]["s_t"][0])}
- Current status: 
    {json.dumps(self.transition_trajectory[-1]["s_t"][1])}
- Previous actions and rationale: 
    {json.dumps([self.transition_trajectory[i]["a_t"] for i in range(len(self.transition_trajectory) - 1)][-HISTORY_SIZE:])}
- Rewards received so far: 
    {self._total_reward}

You task is to:
- propose the top 3 actions to execute at next step based on the current observation and status.
- provide the rationale you think each action is a good choice.
- predict the consequences of executing each action you propose.
- based on the consequence, select the best action to execute next.
- provide the justification for your choice.

Note: You can only select actions from the available actions: {json.dumps(self._available_actions)}.
Note: Avoid unnecessary crafting and placement if the items are within reachable distance.
Note: You should craft as many different tools as soon as possible to increase the chance of reaching the final goal.

Please format your response in the following format: \n{json.dumps(action_format_cot, indent=4)}
"""
        messages.append({"role": "user", "content": game_info})
        if self.agent_type == "plain":
            prompt = prompt_plain
        elif self.agent_type == "cot":
            prompt = prompt_cot
        else:
            raise ValueError("Invalid agent type")
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
    actor = Actor(step_budget=1000, agent_type="plain", seed=43)  # plain or cot
    try:
        actor.run()
    except KeyboardInterrupt:
        print("\nterminated by user, saving log...")
        actor.save_log()

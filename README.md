## Deterministic Crafter
Modified from Danijar Hafner's env, making it deterministic. 

### Simple LLM-based agent
```sh
pip install --upgrade .  # Install Crafter
pip install pygame   # Needed for human interface
cp llm.cfg.template llm.cfg  # Then fill in llm.cfg your LLM endpoint info; don't worry, there's a git ignore
python3 crafter.run_cot_agent  # Start the game
```
TODO: add history info. 

### Collect trajectory data
```sh
pip install --upgrade .  # Install Crafter
pip install pygame   # Needed for human interface
python3 crafter.collect_trajectory      # Start the game
```
A `data` folder will be created, after every game, a trajectory file (with the random seed being used) with random name will be saved into that folder. 

<details>
<summary>Keyboard mapping (click to expand)</summary>

| Key | Action |
| :-: | :----- |
| WASD | Move around |
| SPACE| Collect material, drink from lake, hit creature |
| TAB | Sleep |
| T | Place a table |
| R | Place a rock |
| F | Place a furnace |
| P | Place a plant |
| 1 | Craft a wood pickaxe |
| 2 | Craft a stone pickaxe |
| 3 | Craft an iron pickaxe |
| 4 | Craft a wood sword |
| 5 | Craft a stone sword |
| 6 | Craft an iron sword |

</details>

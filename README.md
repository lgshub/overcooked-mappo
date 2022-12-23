# overcooked-mappo
We apply MAPPO in Overcooked environment using Gym API.

# How to check the results?
Set the 'layout_opt' to be 1~5 at the beginning of 'main.py', which correspond to the 5 layouts we've trained.
1. cramped room
2. asymmetric advantages
3. coordination ring
4. counter circuit
5. forced coordination

# Key files
The featurization coding and shaped reward are given in 'overcooked-mappo/overcooked_ai_py/mdp/overcooked_mdp.py'.

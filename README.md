# overcooked-mappo
We apply MAPPO in Overcooked environment using Gym API.
Each strategy we found takes 30~40 steps to deliver a soup in each layout.

# How to check the results?
Run 'main.py' and the results are rendered in console.
Set the 'layout_opt' to be 1~5 at the beginning of 'main.py', which correspond to the 5 layouts we've trained.
1. cramped room
2. asymmetric advantages
3. coordination ring
4. forced coordination
5. counter circuit

# Key files
Featurization coding and shaped reward are given in 'overcooked-mappo/overcooked_ai_py/mdp/overcooked_mdp.py'.

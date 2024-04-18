# ToolsRL
A research exploration into reinforcement learning agents' ability to learn how to use arbitrary tools to problem-solve and accomplish tasks.
To set up:

    conda create -n tools-rl python=3.10
    conda activate tools-rl
    bash setup.sh

To test the environment without reinforcement learning:

    python3 ./tests/tool_sandbox.py --tool <toolname> --goal <goalname>
    
where `toolname` is one of the tools in `configurations/pz_ppo_config.yaml` (or a specified config file).

To test the environment with reinforcement learning:

    python3 ./tests/stable_baselines_ppo.py --train True --policy <cnn|mlp> --steps <num_steps> --deterministic False

You should be able to train and retrieve a model from this, of which you can observe in action at the end of training.



Important notes:

 - All independent work for this project is outside the `references/` folder.
   The `references/` folder is the only folder in this project that I did not code up.
   I found these resources online to help guide my development throughout the project.

 - You can edit the `configurations/pz_ppo_config.yaml` to use custom tools and goals
   to test in the sandbox and use in the reinforcement learning environment.

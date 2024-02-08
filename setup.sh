poetry install
pip3 install "ray[rllib]"
pip3 install torch==2.0.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 --no-cache-dir
pip3 install stable-baselines3 # Poetry strangely takes forever with this one, so we install it here...

"""Kerchak's reinforcement-learning pipeline: PPO (PyTorch) on the MuJoCo model of the bare gorilla-human hybrid
frame (``labs/legged/designs/space_unit_robots.bare``), command-conditioned knuckle-walking.

Kept apart from the robot's design and control code (``labs/legged``): this package only reads the robot model
(``robot_model``), the programmed gaits (``reference``) and Chiron's terrains and servo law (``vegeta.chiron``).

    training/kerchak_rl/
        config.py         the configuration (dataclasses, YAML, deep-merge, snapshots)
        robot_model.py    the MuJoCo model from the design, the joint and actuator tables, the limits, the contact geoms
        commands.py       the command interface shared by learned and programmed controllers
        pd.py             the PD controller: policy offsets -> joint targets -> torques within the servo law
        terrain.py        flat and uneven terrains (Chiron's)
        rewards.py        the weighted reward terms
        randomization.py  domain randomization
        reference.py      the programmed knuckle walk as a reference trajectory
        env.py            the Gymnasium-compatible environment
        vec_env.py        vectorised environments (in-process and subprocess)
        controllers.py    the controller interface: programmed gaits and learned policies alike
        curriculum.py     the training stages
        ppo/              the actor-critic, the PPO update, the runner (checkpoints, TensorBoard, seeds)
        export.py         the actor to ONNX
        scripts/          train, evaluate, infer, visualize, export_onnx, compare_gaits, teleop
"""
__version__ = "0.1.0"

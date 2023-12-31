CONFIG = {
    "MAX_FRAME": 1_000_000,
    "SYNC_RATE": 1000,
    "REPLAY_SIZE": 100_000,
    "LEARNING_RATE": 0.0001,
    "GAMMA": 0.99,
    "BATCH_SIZE": 32,
    "TAU": 0.005,
    "ENV_NAME": "SpaceInvadersNoFrameskip-v4",
    "OBS_SHAPE": (84, 84),
    "CHKPT_DIR": "./checkpoint"
}
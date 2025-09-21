import glob
import random

import chess
import lightning as L
import torch

from .actor_critic import TrainingLoop


def get_training_loop():
    loop = TrainingLoop(128, "cpu")
    model_paths = glob.glob("models/actor_critic/*")
    model_paths.sort()

    if model_paths != []:
        loop.load_checkpoint(model_paths[-1])

    return loop


def save(loop: TrainingLoop, index):
    if index % 200 == 0 or (index < 100 and index % 25 == 0):
        loop.save_checkpoint(f"models/actor_critic/{loop.episodes:010d}.ckpt")


def main():
    loop = get_training_loop()
    loop.train(1000_000, save=save)


if __name__ == "__main__":
    main()

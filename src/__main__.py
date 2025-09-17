import glob

import chess
import lightning as L
import torch

from .policy_gradients import ChessPolicy, TrainingLoop


def get_training_loop():
    model_paths = glob.glob("models/policy_gradients/model_*.ckpt")
    model_paths.sort()

    if model_paths == []:
        return TrainingLoop(ChessPolicy(32), 50)

    return torch.load(model_paths[-1])


def main():

    loop = get_training_loop()

    for i in range(1000_000):

        loop.train_once()

        if i % 1000 == 0:
            torch.save(loop, f"models/policy_gradients/{loop.games:06d}.ckpt")


if __name__ == "__main__":
    main()

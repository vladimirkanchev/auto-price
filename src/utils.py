"""'Helper functions for face classification algorithm."""

import pathlib
import random
from typing import Dict, List, Tuple
import os
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import TensorDataset
from torchinfo import summary


def save_model(
    model: torch.nn.Module,
    target_dir_path: pathlib.PosixPath,
    model_name: str,
    optimizer: torch.optim.Optimizer,
    trained_model_info: Dict,
):
    """Save a PyTorch model to a target directory.

    Args:
      model: A target PyTorch model to save.
      target_dir_path: A directory for saving the model to.
      model_name: A filename for the saved model. Should include
          either ".pth" or ".pt" as the file extension.
      optimizer: An object holding selected  optimization algorithm
          and its parameters which later will update model weights
      val_loss: A current validation loss of the saved model
      val_acc: A current validation accuracy of the saved model
      epoch: A epoch of training when the model is saved

    Example usage:
      save_model(model=model_0,
                 target_dir="models",
                 model_name="05_going_modular_tingvgg_model.pth",
                 optimizer= optim.SGD(model.parameters(), lr=0.01,
                                      momentum=0.9),
                 val_loss = curr_val_loss,
                 val_acc = curr_val_acc,
                 epoch = 20)
    """
    # Create target directory
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict(),
            "val_loss": trained_model_info["val_loss"],
            "val_acc": trained_model_info["val_acc"],
            "epoch": trained_model_info["epoch"],
        },
        model_save_path,
    )

    # wandb.save(model_save_path.as_posix())


def load_model(
    model: torch.nn.Module,
    target_dir_path: pathlib.PosixPath,
    model_name: str,
    optimizer: torch.optim.Optimizer,
) -> torch.nn.Module:
    """Load a PyTorch model to a target directory.

    Args:
      model: A target PyTorch model to save.
      target_dir_path: A directory to load the model.
      model_name: A filename for the loaded model. Should include
          either ".pth" or ".pt" as the file extension.
      optimizer: An object holding selected optimization algorithm
          and its parameters.
    Example usage:
      load_model(model=model_0,
                target_dir="models",
                model_name="05_going_modular_tingvgg_model.pth",
                optimizer= optim.SGD(model.parameters(),
                                      lr=0.01, momentum=0.9)
                )
    """
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    model_load_path = target_dir_path / model_name

    checkpoint = torch.load(model_load_path)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    opt = optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.eval()

    return model, opt


def save_test_results(test_results: List,
                      test_dataset: data.Dataset,
                      out_fname='classification_early_submission.csv') -> None:
    """Save classified test images into a csv file."""
    with open(out_fname, "w+", encoding="utf8") as fname:
        fname.write("id,label\n")
        for i in range(len(test_dataset)):
            fname.write(f"{str(i).zfill(6)}.jpg,{test_results[i]}\n")


def set_seed() -> None:
    """Set all random parameters to deterministic values."""
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


def read_data(data_dir: pathlib.PosixPath, split: str):
    """Load data and labels from a file and construct a tensordataset."""
    filename = f"{split}{'.pt'}"
    x_vector, y_vector = torch.load(os.path.join(data_dir, filename))

    return TensorDataset(x_vector, y_vector)


def show_model_summary(model: torch.nn.Module,
                       input_size: Tuple[int, int, int, int]):
    """Load the model with a batch of images to show model information."""
    summary(model=model, input_size=input_size,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20, row_settings=["var_names"])


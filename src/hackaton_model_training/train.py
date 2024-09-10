import logging
import os
from statistics import mean

import click
import mlflow
import torch
import torchvision
from dotenv import load_dotenv
from torch.onnx import dynamo_export
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from src.hackaton_model_training.data.dataset import RansomSideDataset

load_dotenv()


def train(
    dataset_path: str,
    lr: float,
    save_path: str,
    save_every: int,
    epochs: int,
    model_name: str,
    device: str,
    mlflow_tracking: bool,
):
    if mlflow_tracking:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))

        mlflow.log_params(
            {
                "lr": lr,
                "save_path": save_path,
                "save_every": save_every,
                "epochs": epochs,
                "model_name": model_name,
                "device": device,
            }
        )

    if device != "cpu" and not torch.cuda.is_available:
        raise RuntimeError("GPU not available")

    if not os.path.exists(os.path.join(save_path, model_name)):
        os.makedirs(os.path.join(save_path, model_name))

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(320, 320)),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ]
    )
    dataset = RansomSideDataset(path=dataset_path, transform=transform)

    train_dataset = torch.utils.data.Subset(dataset, range(0, int(len(dataset) * 0.8)))
    val_dataset = torch.utils.data.Subset(dataset, range(int(len(dataset) * 0.8), len(dataset)))

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    model = resnet18()
    model.fc = torch.nn.Sequential(torch.nn.Linear(512, 1), torch.nn.Sigmoid())
    model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5, verbose=True)

    for epoch in range(epochs):
        train_losses, val_losses = [], []

        model.train()
        for bottom_image, side_image, labels in train_dataloader:
            input_image = torch.cat([bottom_image, side_image], dim=1).float()
            input_image, labels = input_image.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_image)
            train_loss = torch.nn.BCEWithLogitsLoss()(outputs, labels.reshape(-1, 1))
            train_loss.backward()
            optimizer.step()

            train_losses.append(train_loss.item())

        model.eval()
        with torch.no_grad():
            for bottom_image, side_image, labels in val_dataloader:
                input_image = torch.cat([bottom_image, side_image], dim=1).float()
                input_image, labels = input_image.to(device), labels.to(device)
                outputs = model(input_image)
                val_loss = torch.nn.BCEWithLogitsLoss()(outputs, labels.reshape(-1, 1))
                scheduler.step(train_loss)

                val_losses.append(val_loss.item())

        if mlflow_tracking:
            mlflow.log_metrics(
                {"train_loss": mean(train_losses), "val_loss": mean(val_losses), "lr": scheduler.get_last_lr()},
                step=epoch,
            )
        logging.info(
            f"Epoch {epoch}, train_loss: {mean(train_losses)}, val_loss: {mean(val_losses)}, lr: {scheduler.get_last_lr()}"
        )

        if (epoch + 1) % save_every == 0:
            logging.info(f"Saving model at epoch {epoch}")
            torch.save(model.state_dict(), os.path.join(save_path, model_name, f"{model_name}_{epoch}.pt"))
            if mlflow_tracking:
                mlflow.log_artifact(os.path.join(save_path, model_name, f"{model_name}_{epoch}.pt"))

    model.to("cpu")
    onnx_program = dynamo_export(model, torch.randn((1, 2, 320, 320)))
    onnx_program.save(os.path.join(save_path, model_name, f"result_{model_name}.onnx"))

    if mlflow_tracking:
        mlflow.log_artifact(os.path.join(save_path, model_name, f"result_{model_name}.onnx"))


@click.command()
@click.option("--dataset_path", type=click.Path(exists=True))
@click.option("--lr", type=float, default=0.001)
@click.option("--save_path", type=click.Path(), default="models")
@click.option("--save_every", type=int, default=1)
@click.option("--epochs", type=int, default=2)
@click.option("--model_name", type=str, default="resnet")
@click.option("--device", type=str, default="cpu")
@click.option("--mlflow_tracking", type=bool, default=False)
def main(
    dataset_path: str,
    lr: float,
    save_path: str,
    save_every: int,
    epochs: int,
    model_name: str,
    device: str,
    mlflow_tracking: bool,
):
    train(
        dataset_path=dataset_path,
        lr=lr,
        save_path=save_path,
        save_every=save_every,
        epochs=epochs,
        model_name=model_name,
        device=device,
        mlflow_tracking=mlflow_tracking,
    )


if __name__ == "__main__":
    main()

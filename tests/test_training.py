from src.hackaton_model_training.train import train
from src.hackaton_model_training.data.dataset import RansomSideDataset
import os
import shutil

def test_train(test_dataset_path: str):
    model_name = "test_resnet"
    
    train(
        dataset_path=test_dataset_path,
        lr=0.001,
        save_path="models",
        save_every=1,
        epochs=2,
        model_name="test_resnet",
        device="cuda",
        mlflow_tracking=False
    )
    
    assert os.path.exists(f"models/{model_name}/{model_name}_0.pt")
    assert os.path.exists(f"models/{model_name}/{model_name}_1.pt")
    assert os.path.exists(f"models/{model_name}/result_{model_name}.onnx")
    
    shutil.rmtree(f"models/{model_name}")
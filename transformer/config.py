from pathlib import Path

def get_config():

    return {
        "batch_size": 8,
        "num_epochs": 5,
        "lr": 10e-4,
        "max_len": 350,
        "d_model": 512,
        "h": 8,
        "N": 6,
        "d_ff": 2048,
        "dropout": 0.1,
        "src_lang": "en",
        "tgt_lang": "no",
        "model_folder": "./model",
        "model_name": "transformer",
        "preload": None,
        "tokenizer_path": "tokenizer_{0}.json",
        "name": "runs/tmodel",
    }

def get_weights_file_path(config, epoch):
    model_folder = Path(config["model_folder"])
    model_base_name = config["model_name"]
    return model_folder / f"{model_base_name}_{epoch}.pt"
import Algorithmia
import json
import os.path
import joblib
import xgboost
import pandas as pd
import hashlib


client = Algorithmia.client()


def load_model_config(config_rel_path="model_config.json"):
    """Loads the model manifest file as a dict. 
    A manifest file has the following structure:
    {
      "model_filepath": Uploaded model path on Algorithmia data collection
      "model_md5_hash": MD5 hash of the uploaded model file
      "model_origin_repo": Model development repository with the Github CI workflow
      "model_origin_ref": Branch of the model development repository related to the trigger of the CI workflow, 
      "model_origin_commit_SHA": Commit SHA related to the trigger of the CI workflow
      "model_origin_commit_msg": Commit message related to the trigger of the CI workflow
      "model_uploaded_utc": UTC timestamp of the automated model upload
    }
    """
    config = []
    config_path = "{}/{}".format(os.getcwd(), (config_rel_path))
    if os.path.exists(config_path):
        with open(config_path) as json_file:
            config = json.load(json_file)
    return config


def load_model(config):
    """Loads the model object from the file at model_filepath key in config dict"""
    model_path = config["model_filepath"]
    model_file = client.file(model_path).getFile().name
    model_obj = joblib.load(model_file)
    return model_file, model_obj


def assert_model_md5(model_file):
    """
    Calculates the loaded model file's MD5 and compares the actual file hash with the hash on the model manifest
    """
    md5_hash = None
    DIGEST_BLOCK_SIZE = 128 * 64
    with open(model_file, "rb") as f:
        hasher = hashlib.md5()
        buf = f.read(DIGEST_BLOCK_SIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(DIGEST_BLOCK_SIZE)
        md5_hash = hasher.hexdigest()
    assert config["model_md5_hash"] == md5_hash
    print("Model file's runtime MD5 hash equals to the upload time hash, great!")


config = load_model_config()
xgb_path, xgb_obj = load_model(config)
assert_model_md5(xgb_path)


# API calls will begin at the apply() method, with the request body passed as 'input'
# For more details, see algorithmia.com/developers/algorithm-development/languages
def apply(input):
    series_input = pd.Series([input])
    result = xgb_obj.predict(series_input)
    return {
        "sentiment": result.tolist()[0],
        "predicting_model_metadata": {
            "model_file": config["model_filepath"],
            "origin_repo": config["model_origin_repo"],
            "origin_commit_SHA": config["model_origin_commit_SHA"],
            "origin_commit_msg": config["model_origin_commit_msg"],
        },
    }


if __name__ == "__main__":
    algo_input = "It was a great purchase"
    print(f"Calling algorithm apply() func with input: {algo_input}")

    algo_result = apply(algo_input)
    print(algo_result)

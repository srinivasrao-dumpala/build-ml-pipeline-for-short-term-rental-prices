import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            #  CLI : mlfow run . -P steps="download"
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            # CLI : mlfow run . -P steps="basic_cleaning"
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src" , "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact":"sample.csv:latest",
                    "output_artifact":"clean_sample.csv",
                    "output_type":"clean_sample Data",
                    "output_description":" Bsaic Cleaning from sample data",
                    "min_price":config['etl']['min_price'],
                    "max_price":config['etl']['max_price']
                },
            )

        if "data_check" in active_steps:
            # CLI : mlfow run . -P steps="data_check"
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src" , "data_check"),
                "main",
                parameters={
                    "csv":"clean_sample.csv:latest",
                    "ref":"clean_sample.csv:reference",
                    "kl_threshold":config['data_check']['kl_threshold'],
                    "min_price":config['etl']['min_price'],
                    "max_price":config['etl']['max_price']    
                },
            )

        if "data_split" in active_steps:
            # CLI : mlfow run . -P steps="data_split"
            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                "main",
                version='main',
                parameters={
                    "input":"clean_sample.csv:latest",
                    "test_size":config['modeling']['test_size'],
                    "random_seed":config['modeling']['random_seed'],
                    "stratify_by":config['modeling']['stratify_by']
                }
            )

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step
            # mlflow run . -P steps="train_random_forest"
            # mlflow run . -P hydra_options="modeling.max_tfidf_features10,15,30 modeling.random_forest.max_feature=0.1,0.33,0.5,0.75,1.0 -m"
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src" , "train_random_forest"),
                "main",
                parameters= {
                    "val_size":config['modeling']['val_size'],
                    "random_seed":config['modeling']['random_seed'],
                    "stratify_by":config['modeling']['stratify_by'],
                    "max_tfidf_features":config['modeling']['max_tfidf_features'],
                    "rf_config" : rf_config,
                    "trainval_artifact":"trainval_data.csv:latest",
                    "output_artifact":"random_forest_export"
                }
            )

        if "test_regression_model" in active_steps:
            # CLI : mlflow run . -P steps=test_regression_model
            _ = mlflow.run(
                f"{config['main']['components_repository']}/test_regression_model",
                "main",
                version='main',
                parameters = {
                    "mlflow_model":"random_forest_export:prod",
                    "test_dataset":"test_data.csv:latest"
                }
            )

if __name__ == "__main__":
    go()

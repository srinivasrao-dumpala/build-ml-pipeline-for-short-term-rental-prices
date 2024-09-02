#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning
and exporting the result to new artifact for next component
"""

import os
import argparse
import logging

import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    project_name = "nyc_airbnb"
    if "WANDB_PROJECT" in os.environ:
        project_name = os.environ["WANDB_PROJECT"]
    # Initialize W&B Project and run
    run = wandb.init(project = project_name , job_type = 'basic_cleaning')
    run.config.update(args)

    logger.info("Downloading the input artifact %s", args.input_artifact)
    input_artifact = run.use_artifact(args.input_artifact)
    input_artifact_path = input_artifact.file()

    df = pd.read_csv(input_artifact_path)
    logger.info("cleaning raw dataset")
    # Price range modification
    price_range_index = df['price'].between(int(args.min_price), int(args.max_price))
    df = df[price_range_index].copy()
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    df = df.drop_duplicates().reset_index(drop=True)

    #Temporary File
    file_name = "clean_sample.csv"
    df.to_csv(file_name , header=True, index= False)

    # Create W&B Artifact for this run
    artifact = wandb.Artifact(
        name = args.output_artifact,
        type = args.output_type,
        description = args.output_description
    )

    # Add file  to artifact
    artifact.add_file(file_name)
    logger.info("Cleaning is completed and logging artifact %s" , args.output_artifact)
    run.log_artifact(artifact)
    logger.info("Artifact is logged for job %s" , "basic_cleaning" )

    # Remove temporary file
    os.remove(file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=" Basic cleainig Aruments")
    
    parser.add_argument(
       "--input_artifact",
       type=str,
       help="Raw Dataset File Name for the basic cleaning",
       required= True
    )

    parser.add_argument(
       "--output_artifact",
       type=str,
       help="Cleaned dataset File Name",
       required= True
    )

    parser.add_argument(
       "--output_type",
       type=str,
       help="Type of the output Artifact",
       required= True
    )

    parser.add_argument(
       "--output_description",
       type=str,
       help="Short and readable description of output Artifact",
       required= True
    )

    parser.add_argument(
       "--min_price",
       type=str,
       help="Minimum price in USD allowed for cleaning",
       required= True
    )

    parser.add_argument(
       "--max_price",
       type=str,
       help="Maximum Price in USD allowed for cleaning",
       required= True
    )
    args = parser.parse_args()
    go(args)

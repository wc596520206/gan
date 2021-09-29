# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
from process_data import ProcesssData
from model.dcgan import DcGan
from train import Train

if __name__ == "__main__":
    # Read the input information by the user on the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="config path of model",
                        default=r"config\dcgan.json")

    args = parser.parse_args()
    model_file = args.config_file
    with open(model_file, "r", encoding="UTF-8") as fr:
        config = json.load(fr)

    log_path = config["global"]["log_path"]
    if log_path:
        if not os.path.exists(os.path.dirname(log_path)):
            os.makedirs(os.path.dirname(log_path))
        logger = logging.getLogger("对抗网络")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_path, encoding="UTF-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


    processData = ProcesssData(config)
    dcgan = DcGan(config=config)
    train = Train(config,processData)
    train.build_model(dcgan)
    train.train()




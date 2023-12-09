# Import modules
import os
import sys
import tqdm
import torch
import random
import logging
import argparse
import numpy as np


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.DEBUG):
        super().__init__(level)
        self.stream = sys.stdout

    def flush(self):
        self.acquire()
        try:
            if self.stream and hasattr(self.stream, "flush"):
                self.stream.flush()
        finally:
            self.release()

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg, self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit, RecursionError):
            raise
        except Exception:
            self.handleError(record)

def write_log(logger, message):
    if logger:
        logger.info(message)

def set_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def path_check(args):
    # Preprocessing Path Checking
    #(args.preprocess_path, args.task, args.data_name, args.tokenizer)
    if not os.path.exists(args.preprocess_path):
        os.makedirs(args.preprocess_path, exist_ok=True)

    if not os.path.exists(os.path.join(args.preprocess_path, args.data_name)):
        os.makedirs(os.path.join(args.preprocess_path, args.data_name), exist_ok=True)

    if not os.path.exists(os.path.join(args.preprocess_path, args.data_name, args.encoder_model_type)):
        os.makedirs(os.path.join(args.preprocess_path, args.data_name, args.encoder_model_type), exist_ok=True)

    # Model Checkpoint Path Checking
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path, exist_ok=True)

    if not os.path.exists(os.path.join(args.model_save_path, args.data_name)):
        os.makedirs(os.path.join(args.model_save_path, args.data_name), exist_ok=True)

    if not os.path.exists(os.path.join(args.model_save_path, args.data_name, args.encoder_model_type)):
        os.makedirs(os.path.join(args.model_save_path, args.data_name, args.encoder_model_type), exist_ok=True)

    # Testing Results Path Checking
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path, exist_ok=True)
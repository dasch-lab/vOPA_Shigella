# imports
import pandas as pd
import torch
import torchvision.transforms as transforms
import torch.utils.data as data

# built-in imports
import os
import datetime
import json
import socket
import logging
logger = logging.getLogger("vOPA_logger")

# module imports
from utils.dataset import Dataset
from utils.custom_transforms import get_transforms
from utils.utils import write_results, get_model, get_labels
from utils.logger import set_logger
from utils.ml_utils import test, custom_collate
from utils.parsers import inference_parser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != "cpu":
    device_name = torch.cuda.get_device_name(0)
    device_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"Using device: {device_name} with {round(device_vram, 2)} GB of VRAM")
else:
    logger.info(f"Using device: {device}")


def main(args):
    _, test_transforms = get_transforms(args)

    test_df = pd.read_csv(args.test_set, sep='\t')
    logger.info(f'Loading test data from {args.test_set}')
    logger.info(f'Test set has {test_df.shape[0]} images')

    test_df = get_labels(test_df, args.concentration_th)

    # create dataloader
    test_dataset = Dataset(df=test_df, transforms=test_transforms, tr_channels=args.channels, dataset_path=args.dataset_path)
    test_dataloader = data.DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=custom_collate)

    test_metrics, df, patch_df = test(model, test_dataloader, device, args, 'test')
    logger.info(
        f'Test set metrics: Accuracy: {test_metrics["test/accuracy"]:.3f}\tPrecision: {test_metrics["test/precision"]:.3f}\tRecall: {test_metrics["test/recall"]:.3f}\tF1: {test_metrics["test/f1_score"]:.3f}')
    logger.info(f'Confusion matrix:\n{test_metrics["test/tp"]}\t{test_metrics["test/fn"]}\n{test_metrics["test/fp"]}\t{test_metrics["test/tn"]}\n')

    write_results(args.output, test_metrics, df, patch_df, test_df, args.channels, args.concentration_th)


if __name__ == '__main__':

    args = inference_parser()

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    time = datetime.datetime.now().strftime("%H-%M-%S")
    logger_file_name = f'{args.output}/inference_{date}_{time}.log'
    set_logger(logger_file_name)

    # check if output directory exists
    if not os.path.exists(os.path.join(args.output, 'plots')):
        logger.info(f'Creating output directory {os.path.join(args.output, "plots")}')
        os.makedirs(os.path.join(args.output, 'plots'))


    logger.info(f'Loading checkpoint {args.checkpoint}')
    state_dict = torch.load(args.checkpoint)
    args.model = state_dict['architecture']
    args.channels = state_dict['train_channels']
    args.patch_size = state_dict['patch_size']
    args.concentration_th = state_dict['concentration_th']
    model = get_model(args.model, len(args.channels), device)
    model.load_state_dict(state_dict['model'])

    arg_line = '\nStarting with arguments:\n'
    # get longest argument name
    max_len = max([len(arg) for arg in vars(args)])
    for arg in vars(args):
        arg_line += f"{arg:{max_len}} : {getattr(args, arg)}\n"
    arg_line += '\n'
    logger.info(arg_line)

    main(args)

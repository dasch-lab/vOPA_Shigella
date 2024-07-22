# built-in imports
import argparse
import os

import logging
logger = logging.getLogger("vOPA_logger")


def training_parser():
    parser = argparse.ArgumentParser()

    # ---- parameters for the training ----
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patch_size', type=int, default=540, help='patch size')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
    parser.add_argument('--n_cv_splits', type=int, default=4, help='Number of cross-validation splits')
    parser.add_argument('--test_batch_size', type=int, default=8, help='batch size for inference (lower then training since the images are divided in patches)')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--virtual_batch_size', type=int, default=64, help='virtual batch size')
    
    # ---- misc parameters ----
    parser.add_argument('--training_dataset', type=str, help='name of training dataset', required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=6, help='number of workers for the dataloader')
    parser.add_argument('--wandb', type=int, choices=[0, 1], help='use wandb for logging', default=1)
    parser.add_argument('--wandb_entity', type=str)
    parser.add_argument('--wandb_project', type=str)
    parser.add_argument('--results', type=str, help='path to save the results', required=True)
    parser.add_argument('--fast', choices=[0, 1], default=1, help='If = 0 compute metrics also on training set at the end of the epoch')

    # ---- parameters changed in paper experiments ----
    parser.add_argument('--split', type=int, choices=[1, 2, 3, 4], help='4 splits that correspond to the 4 plates, or to four random test splits', default=1)
    parser.add_argument('--split_type', type=str, choices=['batch_stratified', 'batch_separated', 'reduced', 'single_plate'], default='batch_separated', help='Which type of split to use')
    parser.add_argument('--model', type=str, help='model architecture', default='efficientnet_s',
                        choices=['efficientnet_s', 'efficientnet_m',
                                 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                 'densenet121', 'densenet161'])
    parser.add_argument('--concentration_th', default=0.312, type=float, help='Min concentration for the positive class')
    parser.add_argument('--channels', type=str, default='CellMask,mCherry,A488,DAPI')


    args = parser.parse_args()

    args.channels = args.channels.split(',')

    args.results = os.path.join('results', 'training', args.results)

    # checks
    # virtual batch size must be a multiple of batch size
    assert args.virtual_batch_size % args.batch_size == 0, "virtual batch size must be a multiple of batch size"
    # patience should be less than epochs
    assert args.patience < args.epochs, "Patience should be less than epochs"

    # check if the results folder exists
    if not os.path.exists(args.results):
        logger.info(f"Creating folder {args.results}")
        os.makedirs(args.results)
        os.makedirs(os.path.join(args.results, 'models'))
        os.makedirs(os.path.join(args.results, 'test_res'))
        for i in range(1, 5):
            os.makedirs(os.path.join(args.results, 'test_res', f'fold_{i}'))

    return args


def inference_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set', type=str, help='path to dataframe with info about the plate', required=True)
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint', required=True)
    parser.add_argument('--output', type=str, help='path where to save output file', required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--test_batch_size', type=int, default=16, help='batch size for inference (lower then training since the images are divided in patches)')
    parser.add_argument('--num_workers', type=int, default=6, help='number of workers for the dataloader')

    args = parser.parse_args()

    args.output = os.path.join('results', 'inference', args.output)

    return args

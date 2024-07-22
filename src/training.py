# module imports
from utils.dataset import Dataset
from utils.parsers import training_parser
from utils.logger import set_logger
from utils.custom_transforms import SplitPatchTransform, get_transforms
from utils.ml_utils import EarlyStopping, custom_collate, train, test
from utils.utils import wandb_setup, get_model, write_results, create_splits

import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from torchinfo import summary
import wandb
import socket

import time
import json
import os
import logging
logger = logging.getLogger("vOPA_logger")


def main(args):
    # get the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cpu":
        device_name = torch.cuda.get_device_name(0)
        device_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"Using device: {device_name} with {round(device_vram, 2)} GB of VRAM")
    else:
        logger.info(f"Using device: {device}")

    train_transforms, test_transforms = get_transforms(args)

    cv_accuracies = []
    for cv_split in range(1, args.n_cv_splits + 1):
        train_df, val_df, test_df = create_splits(args, cv_split)
        # reset indeces
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        if args.wandb:
            wandb_setup(args)
        else:
            os.environ['WANDB_MODE'] = 'disabled'
            os.environ['WANDB_DISABLED'] = 'true'
            wandb.init(mode="disabled")

        train_dataset = Dataset(df=train_df, transforms=train_transforms, tr_channels=args.channels, dataset_path=args.dataset_path)
        val_dataset = Dataset(df=val_df, transforms=test_transforms, tr_channels=args.channels, dataset_path=args.dataset_path)
        test_dataset = Dataset(df=test_df, transforms=test_transforms, tr_channels=args.channels, dataset_path=args.dataset_path)
        logger.info(f"Dataset split into {len(train_dataset)} train samples, {len(val_dataset)} validation samples and {len(test_dataset)} test samples.")

        train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_dataloader = data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate)
        test_dataloader = data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate)


        model = get_model(args.model, len(args.channels), device)

        # create optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)

        # create loss function and weight differently the two classes since they may be unbalanced
        n_negative = train_df['label'].value_counts()[0]
        n_positive = train_df['label'].value_counts()[1]
        total = n_negative + n_positive
        negative_weight = total / n_negative
        positive_weight = total / n_positive
        class_weights = torch.tensor([negative_weight, positive_weight], dtype=torch.float)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        start_time = time.time()
        for epoch in range(args.epochs):
            logger.info(f"Epoch:\t{epoch+1}")
            # train the model
            train_loss = train(model, train_dataloader, loss_fn, optimizer, device, args)
            scheduler.step()
            metrics = {'epoch': epoch + 1}

            # metrics validation set
            val_metrics, _, _ = test(model, val_dataloader, device, args, 'validation', loss_fn=loss_fn)
            if not args.fast:
                # metrics on training set
                train_metrics, _, _ = test(model, train_dataloader, device, args, 'train', loss_fn=loss_fn, train=True)
                metrics = {**metrics, **val_metrics, **train_metrics}
            else:
                metrics = {**metrics, **val_metrics}

            # save the first model anyway
            if epoch == 0:
                state_dict = {"model": model.state_dict(), "train_channels": args.channels, 
                "architecture": args.model, "patch_size": args.patch_size, "concentration_th": args.concentration_th}
                torch.save(state_dict, f"{args.results}/models/fold{cv_split}.pt")

            # save model when validation loss decreases
            early_stopping(val_metrics['validation/loss'])
            if early_stopping.counter == 0:
                state_dict = {"model": model.state_dict(), "train_channels": args.channels,
                "architecture": args.model, "patch_size": args.patch_size, "concentration_th": args.concentration_th}
                torch.save(state_dict, f"{args.results}/models/fold{cv_split}.pt")
            
            # if patience is reached (and has value != 0) stop the training
            if early_stopping.early_stop and args.patience != 0:
                logger.info("Early stopping")
                wandb.log(metrics)
                break

            logger.debug(metrics)
            wandb.log(metrics)

            # clear the cache after each epoch
            torch.cuda.empty_cache()

        stop_time = time.time()
        train_time = stop_time - start_time
        # convert to readable format hh:mm:ss
        train_time = time.strftime("%H:%M:%S", time.gmtime(train_time))

        logger.info(f"Training finished for fold {cv_split}")
        logger.info(f"It took {train_time} seconds to train the model on device {device_name} with {device_vram:.2f} GB of VRAM")

        # load best model and test on test set
        logger.info(f"Loading best model for fold {cv_split}")
        inference_state_dict = torch.load(f"{args.results}/models/fold{cv_split}.pt")
        model.load_state_dict(inference_state_dict['model'])
        test_metrics, df, patch_df = test(model, test_dataloader, device, args, 'test', loss_fn=loss_fn)

        file_name = os.path.join(args.results, 'test_res', f'fold_{cv_split}')
        write_results(file_name, test_metrics, df, patch_df, test_df, args.channels, args.concentration_th)
        metrics = {**metrics, **test_metrics}

        wandb.log(test_metrics)
        logger.info(f"\n----------test results------------\n{test_metrics}\n")

        wandb.finish()

        cv_accuracies.append(test_metrics['test/accuracy'])

        # clear the cache
        torch.cuda.empty_cache()

    logger.info(f"CV accuracies: {cv_accuracies}")
    # only keep the best model
    best_fold = cv_accuracies.index(max(cv_accuracies)) + 1
    for i in range(1, args.n_cv_splits + 1):
        if i != best_fold:
            logger.info(f"Removing model for fold {i}")
            os.remove(f"{args.results}/models/fold{i}.pt")


if __name__ == "__main__":
    args = training_parser()

    file_name = f'{args.results}/training.log'
    set_logger(file_name)

    arg_line = '\nStarting with arguments:\n'
    # get longest argument name
    max_len = max([len(arg) for arg in vars(args)])
    for arg in vars(args):
        arg_line += f"{arg:{max_len}} : {getattr(args, arg)}\n"
    arg_line += '\n'
    logger.info(arg_line)

    main(args)

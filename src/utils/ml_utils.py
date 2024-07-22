# imports
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb

# built-in imports
import os
import logging
logger = logging.getLogger("vOPA_logger")


def custom_collate(batch):
    """Custom collate function to handle the list of patches"""

    # get the list of patches
    images = [item['image'] for item in batch]
    # get the number of patches
    n_patches = torch.tensor([image.shape[0] for image in images])
    # get the labels
    labels = torch.tensor([item['label'] for item in batch])

    # get metadata
    experiments = [item['experiment'] for item in batch]
    rows = [item['row'] for item in batch]
    cols = [item['column'] for item in batch]
    fovs = [item['fov'] for item in batch]

    # stack the images
    images = torch.cat(images, dim=0).reshape(-1, images[0].shape[-3], images[0].shape[-2], images[0].shape[-1])

    # stack the labels and all metadata
    labels = torch.repeat_interleave(labels, n_patches)

    experiments = [experiment for experiment, num_patches in zip(experiments, n_patches) for _ in range(num_patches)]
    rows = [row for row, num_patches in zip(rows, n_patches) for _ in range(num_patches)]
    cols = [col for col, num_patches in zip(cols, n_patches) for _ in range(num_patches)]
    fovs = [fov for fov, num_patches in zip(fovs, n_patches) for _ in range(num_patches)]

    patch_names = [f"p{patch_number+1:02d}" for num_patches in n_patches for patch_number in range(num_patches)]

    collated_batch = {'image': images, 'label': labels, 'patch_names': patch_names, 'experiment': experiments,
                      'row': rows, 'column': cols, 'fov': fovs}

    return collated_batch


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.min_val_loss = np.Inf
        self.early_stop = False

    def __call__(self, val_loss):
        if ((val_loss + self.min_delta) < self.min_val_loss):
            if self.verbose:
                logger.info(f"Validation loss decreased ({self.min_val_loss:.6f} --> {val_loss:.6f}).  Saving model ...")
            self.min_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"Validation loss did not decrease {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def train(model, dataloader, loss_fn, optimizer, device, args):
    model.train()
    epoch_loss = 0
    virtual_iters = args.virtual_batch_size // args.batch_size
    optimizer.zero_grad()
    with tqdm(enumerate(dataloader), total=len(dataloader)) as pbar:
        for i, batch in pbar:
            # clear cache before each batch
            torch.cuda.empty_cache()
            # get the inputs; data is a list of [inputs, labels]
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device).long()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()

            # gradient accumulation
            if (i + 1) % virtual_iters == 0:
                optimizer.step()
                # zero the parameter gradients
                optimizer.zero_grad()

            pbar.set_description(f"Loss: {loss.item():.3f}")
            epoch_loss += loss.item()
            wandb.log({'train/current_train_loss': loss.item()})

    epoch_loss /= len(dataloader)
    return epoch_loss


@torch.no_grad()
def test(model, dataloader, device, args, set_name, loss_fn=None, train=False):
    """Function for testing the model in evaluation mode"""
    model.eval()
    if loss_fn:
        test_loss = 0
    else:
        test_loss = np.nan

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    image_df = pd.DataFrame(columns=['label', 'prediction', 'experiment', 'row', 'column', 'fov', ])
    patch_df = pd.DataFrame(columns=['label', 'prediction', 'experiment', 'row', 'column', 'fov', 'patch'])

    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # clear cache before each batch
        torch.cuda.empty_cache()
        # get the inputs; data is a list of [inputs, labels]
        inputs = batch['image'].to(device)
        labels = batch['label'].to(device)

        rows = batch['row']
        cols = batch['column']
        fovs = batch['fov']
        experiments = batch['experiment']

        if not train:
            patch_names = batch['patch_names']

        outputs = model(inputs)

        if loss_fn:
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
        # get the predictions
        _, preds = torch.max(outputs, 1)

        if not train:
            if patch_df.empty:
                patch_df = pd.DataFrame({'patch': patch_names, 'label': labels.cpu().numpy(), 'prediction': preds.cpu().numpy(), 'experiment': experiments,
                                        'row': rows, 'column': cols, 'fov': fovs, 'patch': patch_names})
            else:
                patch_df = pd.concat([patch_df, pd.DataFrame({'patch': patch_names, 'label': labels.cpu().numpy(), 'prediction': preds.cpu().numpy(),
                                     'experiment': experiments, 'row': rows, 'column': cols, 'fov': fovs, 'patch': patch_names})])

            # label, experiment, row, column, fov are the same for all patches, take the ones at the index with patch_name == 'p01'
            # this way it works even if the number of patches is different for each image
            first_patches = [i for i in range(len(patch_names)) if patch_names[i] == 'p01']
            # predictions have to be image-wise, not patch-wise
            labels = labels[first_patches]
            # divide the predictions so that they are by image
            predictions_by_image = []
            for i in range(len(first_patches) - 1):
                start = first_patches[i]
                end = first_patches[i + 1]
                predictions_by_image.append(preds[start:end])
            predictions_by_image.append(preds[first_patches[-1]:])
            experiments = np.array(experiments)[first_patches]
            rows = np.array(rows)[first_patches]
            cols = np.array(cols)[first_patches]
            fovs = np.array(fovs)[first_patches]

            prs = torch.tensor([], dtype=int).to(device)
            for prediction in predictions_by_image:
                # for the predictions, take the most frequent element of each row
                prs = torch.cat((prs, torch.mode(prediction).values.int().unsqueeze(0)))
   

        if not train:
            # concatenate the dataframe
            if image_df.empty:
                image_df = pd.DataFrame({'label': labels.cpu(), 'prediction': prs.cpu(),
                                        'row': rows, 'column': cols, 'fov': fovs, 'experiment': experiments})
            else:
                image_df = pd.concat([image_df, pd.DataFrame({'label': labels.cpu(), 'prediction': prs.cpu(),
                                     'row': rows, 'column': cols, 'fov': fovs, 'experiment': experiments})])

        # get the true positives, true negatives, false positives and false negatives
        tp += torch.sum((prs == 1) & (labels == 1)).item()
        tn += torch.sum((prs == 0) & (labels == 0)).item()
        fp += torch.sum((prs == 1) & (labels == 0)).item()
        fn += torch.sum((prs == 0) & (labels == 1)).item()

    test_loss /= len(dataloader)
    logger.debug(f"tp: {tp}\ttn: {tn}\tfp: {fp}\tfn: {fn}")
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy = round(accuracy, 3)
    precision = round(precision, 3)
    recall = round(recall, 3)
    f1 = round(f1, 3)
    keys = ['loss', 'accuracy', 'precision', 'recall', 'f1_score', 'tp', 'tn', 'fp', 'fn']
    keys = [f"{set_name}/{key}" for key in keys]
    values = [test_loss, accuracy, precision, recall, f1, tp, tn, fp, fn]
    metrics = dict(zip(keys, values))
    return metrics, image_df, patch_df

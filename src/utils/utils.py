# imports
import pandas as pd
import wandb
from sklearn.model_selection import StratifiedKFold

from models.efficient_net import EfficientNet
from models.res_net import ResNet
from models.dense_net import DenseNet

import logging
logger = logging.getLogger("vOPA_logger")


# built-in imports
import os


def get_labels(df, concentration_th):
    df['label'] = 0
    # no_phagocytic activity -> label 0
    # phagocytic activity and low concentration -> label 0
    # phagocytic activity and high concentration -> label 1
    df['label'] = df.apply(lambda x: 1 if (x['phagocytic_activity'] == 'yes' and x['concentration'] >= concentration_th) else x['label'], axis=1)

    return df


def wandb_setup(args) -> None:

    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    wandb.define_metric("epoch")

    wandb.define_metric("validation/loss", step_metric="epoch")
    wandb.define_metric("validation/accuracy", step_metric="epoch")
    wandb.define_metric("validation/precision", step_metric="epoch")
    wandb.define_metric("validation/recall", step_metric="epoch")
    wandb.define_metric("validation/f1_score", step_metric="epoch")

    wandb.define_metric("train/loss", step_metric="epoch")
    wandb.define_metric("train/accuracy", step_metric="epoch")
    wandb.define_metric("train/precision", step_metric="epoch")
    wandb.define_metric("train/recall", step_metric="epoch")
    wandb.define_metric("train/f1_score", step_metric="epoch")


def write_results(file_name, metrics, df, patch_df, metadata_df, channels, c_th):
    metadata_df = metadata_df.drop(columns='label')
    metadata_df['channel_names'] = [channels] * len(metadata_df)
    # unite df and metadata_df on experiment, row, column, fov
    df = pd.merge(df, metadata_df, on=['experiment', 'row', 'column', 'fov'])
    # unite df and patch_df on experiment, row, column, fov, patch
    patch_df = pd.merge(patch_df, metadata_df, on=['experiment', 'row', 'column', 'fov'])

    df.to_csv(os.path.join(file_name, 'fov.csv'), index=False, sep='\t')
    patch_df.to_csv(os.path.join(file_name, 'patch.csv'), index=False, sep='\t')

    metrics_df = pd.DataFrame(metrics, index=[0])
    col_names = metrics_df.columns
    col_names = [col.split('/')[1] for col in col_names]
    metrics_df.columns = col_names
    metrics_df.drop(columns=['tp', 'tn', 'fp', 'fn', 'loss'], inplace=True)
    metrics_df.to_csv(os.path.join(file_name, 'metrics.csv'), index=False, sep='\t')


def get_model(model_name, n_channels, device, n_classes=2):
    # create model
    if model_name == 'efficientnet_s':
        model = EfficientNet(in_channels=n_channels, out_channels=3, nn_classes=n_classes, size='s').to(device)
    elif model_name == 'efficientnet_m':
        model = EfficientNet(in_channels=n_channels, out_channels=3, nn_classes=n_classes, size='m').to(device)
    elif model_name == 'resnet18':
        model = ResNet(in_channels=n_channels, out_channels=3, nn_classes=n_classes, size='18').to(device)
    elif model_name == 'resnet34':
        model = ResNet(in_channels=n_channels, out_channels=3, nn_classes=n_classes, size='34').to(device)
    elif model_name == 'resnet50':
        model = ResNet(in_channels=n_channels, out_channels=3, nn_classes=n_classes, size='50').to(device)
    elif model_name == 'resnet101':
        model = ResNet(in_channels=n_channels, out_channels=3, nn_classes=n_classes, size='101').to(device)
    elif model_name == 'resnet152':
        model = ResNet(in_channels=n_channels, out_channels=3, nn_classes=n_classes, size='152').to(device)
    elif model_name == 'densenet121':
        model = DenseNet(in_channels=n_channels, size='121', nn_classes=n_classes).to(device)
    elif model_name == 'densenet161':
        model = DenseNet(in_channels=n_channels, size='161', nn_classes=n_classes).to(device)

    return model


def cross_val_splits(n_cv_s, train_wells, y_train_wells, train_df, positive_controls):
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
    tr_index, val_index = list(skf.split(train_wells, y_train_wells))[n_cv_s - 1]
    tr_wells = train_wells[tr_index]
    val_wells = train_wells[val_index]
    # extract the dataframe
    tr_df = train_df[(train_df['experiment'] + train_df['row'] + train_df['column']).isin(tr_wells)]
    val_df = train_df[(train_df['experiment'] + train_df['row'] + train_df['column']).isin(val_wells)]

    return tr_df, val_df


def split_scenario(train_df, n_cv_sp, positive_controls):
    train_wells = (train_df['experiment'] + train_df['row'] + train_df['column']).unique()
    y_train_wells = [train_df['label'].loc[train_df['experiment'] + train_df['row'] + train_df['column'] == well].unique() for well in train_wells]

    # split the train set into train and validation
    tr, val = cross_val_splits(n_cv_sp, train_wells, y_train_wells, train_df, positive_controls)
    return tr, val


def create_splits(args, cv_s):
    # train, test, and validation need to be labeled -> controls.csv
    controls_df = pd.read_csv(args.training_dataset, sep='\t')

    controls_df = get_labels(controls_df.copy(), concentration_th=args.concentration_th)

    positive_controls_df = controls_df[controls_df['phagocytic_activity'] == 'yes']
    positive_controls = positive_controls_df['treatment'].unique()

    plates = sorted(controls_df['experiment'].unique())
    logger.info(f"The dataset has {controls_df.shape[0]} images and {len(plates)} plates:\n{plates}")

    plate = plates[args.split - 1]

    if args.split_type == 'batch_separated':
        logger.info(f"Splitting the dataset using batch separated strategy")
        logger.info(f"Test set: {plate}")
        # ----------- batch separated split -----------
        train_batch_separated = controls_df[controls_df['experiment'] != plate].copy()
        test = controls_df[controls_df['experiment'] == plate].copy()
    
        tr, val = split_scenario(train_batch_separated, n_cv_sp=cv_s, positive_controls=positive_controls)

    elif args.split_type == 'single_plate':
        logger.info(f"Splitting the dataset using single plate strategy")
        logger.info(f"Train set: {plate}")
        # ----------- single plate split -----------
        train_single_plate = controls_df[controls_df['experiment'] == plate].copy()
        
        test = controls_df[controls_df['experiment'] != plate].copy()
        tr, val = split_scenario(train_single_plate, n_cv_sp=cv_s, positive_controls=positive_controls)

    elif args.split_type == 'reduced':
        logger.info(f"Splitting the dataset using reduced strategy")
        logger.info(f"Test set: {plate}")
        # ----------- reduced dataset -----------
        train_reduced = controls_df[controls_df['experiment'] != plate].copy()
        train_reduced = train_reduced.groupby('label').apply(lambda x: x.sample(frac=0.3, random_state=1)).reset_index(drop=True)
        test = controls_df[controls_df['experiment'] == plate].copy()
        tr, val = split_scenario(train_reduced, n_cv_sp=cv_s, positive_controls=positive_controls)
    else:
        # batch stratified splits -> test set is not a plate but from all plates
        skf = StratifiedKFold(n_splits=args.n_cv_splits, shuffle=True, random_state=2)
        wells = (controls_df['experiment'] + controls_df['row'] + controls_df['column']).unique()
        y_wells = [controls_df['label'].loc[controls_df['experiment'] + controls_df['row'] + controls_df['column'] == well].unique() for well in wells]
        train_index, test_index = list(skf.split(wells, y_wells))[args.split - 1]
        train_wells = wells[train_index]
        y_train_wells = [controls_df['label'].loc[controls_df['experiment'] + controls_df['row'] + controls_df['column'] == well].unique() for well in train_wells]
        test_wells = wells[test_index]

        train_batch_stratified = controls_df[(controls_df['experiment'] + controls_df['row'] + controls_df['column']).isin(train_wells)]
        test = controls_df[(controls_df['experiment'] + controls_df['row'] + controls_df['column']).isin(test_wells)]
        tr, val = split_scenario(train_batch_stratified, n_cv_sp=cv_s, positive_controls=positive_controls)

    return tr, val, test

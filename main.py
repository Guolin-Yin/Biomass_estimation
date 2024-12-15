# %%
import torch
from src.tools.dataset import TiffDataset, shuffle_pixel, batch_process, resize_to_224
from src.tools.trainer import Trainer
from src.tools.train_utils import load_checkpoint_cfg, EarlyStopping
from src.model.TFCNN import TFCNN
from src.utils.config import model_params
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--train', default=True, action='store_true', help='train the model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--image_norm', type=float, default=1,
                        help='normalization factor for input images (default: 1000)')
    parser.add_argument('--label_norm', type=float, default=1,
                        help='normalization factor for labels (default: 10)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs for training (default: 1000)')
    parser.add_argument('--model_type', type=str, default='TFCNN',
                        help='model type (default: Terratorch)')
    return parser.parse_args()

args = parse_args()

def norm(image, label):
    return image/args.image_norm, label/args.label_norm

# set up model
if args.model_type == 'TFCNN':
    model = TFCNN(model_params = model_params)
    data_dir = './Running_Dataset/V2/balanced'
    area = 'Portugal'
    processor_fns = [shuffle_pixel, norm]
    fns = batch_process(*processor_fns, mode='concat')
elif args.model_type == 'Terratorch':
    from terratorch.cli_tools import LightningInferenceModel
    config_path = './configs/config.yaml'
    ckpt_path = './biomass_model.ckpt'
    UNUSED_BAND = "-1"
    predict_dataset_bands = [UNUSED_BAND,"BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2",UNUSED_BAND,UNUSED_BAND,UNUSED_BAND,UNUSED_BAND]
    model = LightningInferenceModel.from_config(config_path, ckpt_path, predict_dataset_bands)
    model = model.model
    data_dir = './Running_Dataset/V2/raw_224'
    area = 'Portugal'
    # data_dir = '../granite-geospatial-biomass-datasets/taiga_datasplit'
    # area = 'taiga'
    processor_fns = [resize_to_224, norm]
    
    fns = batch_process(*processor_fns, mode='stack')
# data_dir = './Running_Dataset/V2/balanced'
# data_dir = './resource/granite-geospatial-biomass-datasets/taiga_datasplit'
train_ds = TiffDataset(data_dir, split='train')
val_ds = TiffDataset(data_dir, split='val')
test_ds = TiffDataset(data_dir, split='test')
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=fns)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, collate_fn=fns)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, collate_fn=fns)
print(f"Train dataset length: {len(train_ds)}")
print(f"Val dataset length: {len(val_ds)}")
print(f"Test dataset length: {len(test_ds)}")


# %%    
if args.train:
    # Set up optimizer and loss function for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Different optimizer options:

    # Option 1: SGD with momentum and weight decay
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)

    # Option 2: RMSprop with momentum and centered gradient
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9, centered=True)

    # Option 3: AdamW with weight decay
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Option 4: Adagrad with learning rate decay
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, lr_decay=0.01)

    # Create trainer object
    trainer = Trainer(
        model=model,
        optimizer=optimizer, 
        loss_fn=torch.nn.MSELoss(),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        prefix=f'{area}-lr_{optimizer.param_groups[0]["lr"]}-xnormby{args.image_norm}-ynormby{args.label_norm}-{args.model_type}-opt_{optimizer.__class__.__name__}',
        is_train=True,
        # callbacks = [EarlyStopping(patience=100, min_delta=0)],
        args = args,
        # checkpoint_dir='check_points/Portugal-lr_0.001-xnormby1.0-ynormby1.0-Terratorch/run_2024-12-13->15:28:42',
        # is_transfer_learning=False
    )

    trainer.train(args.epochs, run_test_at_the_end=True)
else:
    test_checkpoint_dir = 'check_points/Portugal-lr_0.001-xnormby1.0-ynormby1.0-Terratorch-opt_Adam/run_2024-12-15->10:23:28'
    # model_params, train_params = load_checkpoint_cfg(test_checkpoint_dir)
    # model = TFCNN(model_params = model_params)
    # print(model_params)
    # print(train_params)
    trainer = Trainer(
        model=model,
        optimizer=None, 
        loss_fn=torch.nn.MSELoss(),
        train_loader=None,
        val_loader=None,
        test_loader=test_loader,
        prefix=None,
        checkpoint_dir= test_checkpoint_dir,
        is_train=False
    )
    trainer.test()
# %%

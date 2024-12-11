# %%
import torch
from src.tools.dataset import TiffDataset, shuffle_pixel, batch_process
from src.tools.trainer import Trainer
from src.tools.train_utils import load_checkpoint_cfg, EarlyStopping
from src.model.TFCNN import TFCNN
from src.utils.config import model_params
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for training (default: 2)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--image_norm', type=float, default=1000,
                        help='normalization factor for input images (default: 1000)')
    parser.add_argument('--label_norm', type=float, default=10,
                        help='normalization factor for labels (default: 10)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs for training (default: 1000)')
    return parser.parse_args()

args = parse_args()

def norm(image, label):
    return image/args.image_norm, label/args.label_norm

processor_fns = [shuffle_pixel, norm]

data_dir = './Running_Dataset/V2/balanced'
train_ds = TiffDataset(data_dir, split='train')
val_ds = TiffDataset(data_dir, split='val')
test_ds = TiffDataset(data_dir, split='test')
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=batch_process(*processor_fns))
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, collate_fn=batch_process(*processor_fns))
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, collate_fn=batch_process(*processor_fns))
print(f"Train dataset length: {len(train_ds)}")
print(f"Val dataset length: {len(val_ds)}")
print(f"Test dataset length: {len(test_ds)}")


# %%
# Set up model (assuming you have a model defined elsewhere)
model = TFCNN(model_params = model_params)

# Set up optimizer and loss function for regression
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = torch.nn.MSELoss()

# Create trainer object
trainer = Trainer(
    model=model,
    optimizer=optimizer, 
    loss_fn=loss_fn,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    prefix=f'Portugal-lr_{optimizer.param_groups[0]["lr"]}-xnormby{args.image_norm}-ynormby{args.label_norm}',
    is_train=True,
    callbacks = [EarlyStopping(patience=20, min_delta=0)],
    args = args
)
# %%
trainer.train(args.epochs, run_test_at_the_end=True)
# %%
# test_checkpoint_dir = 'check_points/pixel_level_regression_normalized/run_2024-11-23->16:33:31'
# model_params, train_params = load_checkpoint_cfg(test_checkpoint_dir)
# model = TFCNN(model_params = model_params)
# print(model_params)
# print(train_params)
# trainer = Trainer(
#     model=model,
#     optimizer=None, 
#     loss_fn=torch.nn.MSELoss(),
#     train_loader=None,
#     val_loader=None,
#     test_loader=test_loader,
#     prefix='pixel_level_regression',
#     checkpoint_dir= test_checkpoint_dir,
#     is_train=False
# )
# trainer.test()
# %%

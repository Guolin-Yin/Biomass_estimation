import torch
import pickle
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .module import Attention, PreNorm, FeedForward
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
import torch.optim as optim
from tensorboardX import SummaryWriter
import pytorch_lightning as pl

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import pandas as pd

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class TSViT(pl.LightningModule):
    """
    Temporal-Spatial ViT5 (used in main results, section 4.3)
    For improved training speed, this implementation uses a (365 x dim) temporal position encodings indexed for
    each day of the year. Use TSViT_lookup for a slower, yet more general implementation of lookup position encodings
    """
    def __init__(self, model_config, train_config):
        super().__init__()
        self._init_model_params(model_config)
        self._init_train_params(train_config)
        # tokenization
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_dim, self.dim),)
        self.to_temporal_embedding_input = nn.Linear(366, self.dim)
        # temporal
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        # spatial
        self.space_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.dim))
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        # output
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2)
        )
        self.to_prob = nn.LogSoftmax(dim=1)
        self.save_hyperparameters()
    def forward(self, x, t):
        B, T, C, H, W = x.shape
        # temporal
        temporal_pos_embedding = self._get_temporal_position_embeddings(t).reshape(B, T, self.dim) #(B, T, self.dim)
        x = self.to_patch_embedding(x)#(B*, T, self.dim)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        # spatial
        x += self.space_pos_embedding#[:, :, :(n + 1)]
        x = self.dropout(x)
        x = self.space_transformer(x)
        # output
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2, self.patch_size**2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return self.to_prob(x)
    def _init_model_params(self, model_config):
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.num_patches = self.num_patches_1d ** 2
        self.patch_dim = (model_config['num_channels']) * self.patch_size ** 2  # -1 is set to exclude time feature
    def _init_train_params(self, train_config):
        '''
        train_config: dict
        {
        'linear_encoder'  : linear_encoder,
        'parcel_loss'     : parcel_loss,
        'run_path'        : run_path,
        'class_weights'   : class_weights,
        'crop_encoding'   : crop_encoding,
        'checkpoint_epoch': checkpoint_epoch
        }
        '''
        self.linear_encoder = train_config['linear_encoder']
        self.parcel_loss = train_config['parcel_loss']

        self.epoch_train_losses = []
        self.epoch_valid_losses = []
        self.avg_train_losses = []
        self.avg_val_losses = []
        self.best_loss = None
        self.learning_rate = train_config['learning_rate']

        num_discrete_labels = len(set(train_config['linear_encoder'].values()))
        self.num_classes = num_discrete_labels
        self.confusion_matrix = torch.zeros((num_discrete_labels, num_discrete_labels)).to('cuda')
        self.class_weights = train_config['class_weights']
        self.checkpoint_epoch = train_config['checkpoint_epoch']

        if train_config['class_weights'] is not None:
            class_weights_tensor = torch.tensor([train_config['class_weights'][k] for k in sorted(train_config['class_weights'].keys())]).cuda()

            if self.parcel_loss:
                self.lossfunction = nn.NLLLoss(ignore_index=0, weight=class_weights_tensor, reduction='sum')
            else:
                self.lossfunction = nn.NLLLoss(ignore_index=0, weight=class_weights_tensor)
        else:
            if self.parcel_loss:
                self.lossfunction = nn.NLLLoss(ignore_index=0, reduction='sum')
            else:
                self.lossfunction = nn.NLLLoss(ignore_index=0)

        self.crop_encoding = train_config['crop_encoding']

        self.run_path = Path(train_config['run_path'])
    def _get_temporal_position_embeddings(self, t):
        t = (t * 365.0001).to(torch.int64)
        t = F.one_hot(t, num_classes=366).to(torch.float32)
        t = t.reshape(-1, 366)
        return self.to_temporal_embedding_input(t)
    # Training part
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        pla_lr_scheduler = {
            'scheduler': lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        factor=0.5,
                                                        patience=4,
                                                        verbose=True),
            'monitor': 'val_loss',
            'reduce_on_plateau': True,
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [pla_lr_scheduler]
    def training_step(self, batch, batch_idx):
        inputs = batch['medians']  # (B, T, C, H, W)
        time = batch['time']  # (B, T)
        label = batch['labels']  # (B, H, W)
        label = label.to(torch.long)

        # Concatenate time series along channels dimension
        # b, t, c, h, w = inputs.size()
        # inputs = inputs.view(b, -1, h, w)   # (B, T * C, H, W)

        pred = self(inputs, time)  # (B, K (the number of categories), H, W)

        if self.parcel_loss:
            parcels = batch['parcels']  # (B, H, W)
            parcels_K = parcels[:, None, :, :].repeat(1, pred.size(1), 1, 1)  # (B, K, H, W)

            # Note: a new masked array must be created in order to avoid inplace
            # operations on the label/pred variables. Otherwise the optimizer
            # will throw an error because it requires the variables to be unchanged
            # for gradient computation

            mask = (parcels) & (label != 0)
            mask_K = (parcels_K) & (label[:, None, :, :].repeat(1, pred.size(1), 1, 1) != 0)

            label_masked = label.clone()
            label_masked[~mask] = 0

            pred_masked = pred.clone()
            pred_masked[~mask_K] = 0

            label = label_masked.clone()
            pred = pred_masked.clone()

            loss = self.lossfunction(pred, label)

            loss = loss / parcels.sum()
        else:
            loss = self.lossfunction(pred, label)

        # Compute total loss for current batch
        loss_aver = loss.item() * inputs.shape[0]

        self.epoch_train_losses.append(loss_aver)

        # torch.nn.utils.clip_grad_value_(self.parameters(), clip_value=10.0)

        return {'loss': loss}
    def training_epoch_end(self, outputs):
        # Calculate average loss over an epoch
        train_loss = np.nanmean(self.epoch_train_losses)
        self.avg_train_losses.append(train_loss)

        with open(self.run_path / "avg_train_losses.txt", 'a') as f:
            f.write(f'{self.current_epoch}: {train_loss}\n')

        with open(self.run_path / 'lrs.txt', 'a') as f:
            f.write(f'{self.current_epoch}: {self.learning_rate}\n')

        self.log('train_loss', train_loss, prog_bar=True)

        # Clear list to track next epoch
        self.epoch_train_losses = []
    def validation_step(self, batch, batch_idx):
        inputs = batch['medians']  # (B, T, C, H, W)
        time = batch['time']  # (B, T)
        label = batch['labels']  # (B, H, W)
        label = label.to(torch.long)

        # Concatenate time series along channels dimension
        # b, t, c, h, w = inputs.size()
        # inputs = inputs.view(b, -1, h, w)   # (B, T * C, H, W)

        pred = self(inputs, time)  # (B, T, C, H, W)

        if self.parcel_loss:
            parcels = batch['parcels']  # (B, H, W)
            parcels_K = parcels[:, None, :, :].repeat(1, pred.size(1), 1, 1)  # (B, K, H, W)

            # Note: a new masked array must be created in order to avoid inplace
            # operations on the label/pred variables. Otherwise the optimizer
            # will throw an error because it requires the variables to be unchanged
            # for gradient computation

            mask = (parcels) & (label != 0)
            mask_K = (parcels_K) & (label[:, None, :, :].repeat(1, pred.size(1), 1, 1) != 0)

            label_masked = label.clone()
            label_masked[~mask] = 0

            pred_masked = pred.clone()
            pred_masked[~mask_K] = 0

            label = label_masked.clone()
            pred = pred_masked.clone()

            loss = self.lossfunction(pred, label)

            loss = loss / parcels.sum()
        else:
            loss = self.lossfunction(pred, label)

        # Compute total loss for current batch
        loss_aver = loss.item() * inputs.shape[0]

        self.epoch_valid_losses.append(loss_aver)

        return {'val_loss': loss}
    def validation_epoch_end(self, outputs):
        # Calculate average loss over an epoch
        valid_loss = np.nanmean(self.epoch_valid_losses)
        self.avg_val_losses.append(valid_loss)

        with open(self.run_path / "avg_val_losses.txt", 'a') as f:
            f.write(f'{self.current_epoch}: {valid_loss}\n')

        self.log('val_loss', valid_loss, prog_bar=True)

        # Clear list to track next epoch
        self.epoch_valid_losses = []
    def test_step(self, batch, batch_idx):
        # print(f"Testing epoch {self.current_epoch}...")
        # print(f"Batch {batch_idx}...")
        inputs = batch['medians']  # (B, T, C, H, W)
        time   = batch['time']  # (B, T)
        label  = batch['labels'].to(torch.long)  # (B, H, W)

        # Concatenate time series along channels dimension
        b, t, c, h, w = inputs.size()


        pred = self(inputs, time).to(torch.long)  # (B, K, H, W)

        # Reverse the logarithm of the LogSoftmax activation
        pred = torch.exp(pred)

        # Clip predictions larger than the maximum possible label
        pred = torch.clamp(pred, 0, max(self.linear_encoder.values()))

        if self.parcel_loss:
            parcels   = batch['parcels']  # (B, H, W)
            parcels_K = parcels[:, None, :, :].repeat(1, pred.size(1), 1, 1)  # (B, K, H, W)

            mask          = (parcels) & (label != 0)
            mask_K        = (parcels_K) & (label[:, None, :, :].repeat(1, pred.size(1), 1, 1) != 0)
            label[~mask]  = 0
            pred[~mask_K] = 0
            pred_sparse   = pred.argmax(axis=1)

            label = label.flatten()
            pred  = pred_sparse.flatten()

            # Discretize predictions
            #bins = np.arange(-0.5, sorted(list(self.linear_encoder.values()))[-1] + 0.5, 1)
            #bins_idx = torch.bucketize(pred, torch.tensor(bins).cuda())
            #pred_disc = bins_idx - 1

            # Update confusion matrix
        self._update_confusion_matrix(label, pred)
    def test_epoch_end(self, outputs):
        def compute_metrics(cm):
            TP = np.diag(cm)
            FP = np.sum(cm, axis=0) - TP
            FN = np.sum(cm, axis=1) - TP
            num_classes = cm.shape[0]
            TN = []
            for i in range(num_classes):
                temp = np.delete(cm, i, 0)    # delete ith row
                temp = np.delete(temp, i, 1)  # delete ith column
                TN.append(sum(sum(temp)))
            specificity = TN/(TN+FP)
            precision = TP/(TP+FP)
            recall = TP/(TP+FN)
            f1 = 2*(precision*recall)/(precision+recall)
            cls_accuracy = (TP+TN)/cm.sum()
            return TP, FP, FN, TN, specificity, precision, recall, f1, cls_accuracy
        self.confusion_matrix = self.confusion_matrix.cpu().detach().numpy()
        self.confusion_matrix = self.confusion_matrix[1:, 1:]  # Drop zero label

        tp, fp, fn, tn, tnr, ppv, tpr, f1, accuracy = compute_metrics(self.confusion_matrix)

        # metric = pd.DataFrame(self.confusion_matrix, index=['a', 'b', 'c', 'd', 'e','f','g','h','I','J',], columns=['a', 'b', 'c', 'd', 'e','f','g','h','I','J',]).da.export_metrics()
        # Export metrics in text file
        metrics_file = self.run_path / f"evaluation_metrics_epoch{self.checkpoint_epoch}.csv"

        # Delete file if present
        metrics_file.unlink(missing_ok=True)

        with open(metrics_file, "a") as f:
            row = 'Class'
            for k in sorted(self.linear_encoder.keys()):
                if k == 0: continue
                row += f',{k} ({self.crop_encoding[k]})'
            f.write(row + '\n')

            row = 'tn'
            for i in tn:
                row += f',{i}'
            f.write(row + '\n')

            row = 'tp'
            for i in tp:
                row += f',{i}'
            f.write(row + '\n')

            row = 'fn'
            for i in fn:
                row += f',{i}'
            f.write(row + '\n')

            row = 'fp'
            for i in fp:
                row += f',{i}'
            f.write(row + '\n')

            row = "specificity"
            for i in tnr:
                row += f',{i:.4f}'
            f.write(row + '\n')

            row = "precision"
            for i in ppv:
                row += f',{i:.4f}'
            f.write(row + '\n')

            row = "recall"
            for i in tpr:
                row += f',{i:.4f}'
            f.write(row + '\n')

            row = "accuracy"
            for i in accuracy:
                row += f',{i:.4f}'
            f.write(row + '\n')

            row = "f1"
            for i in f1:
                row += f',{i:.4f}'
            f.write(row + '\n')

            row = 'weighted macro-f1'
            class_samples = self.confusion_matrix.sum(axis=1)
            weighted_f1 = ((f1 * class_samples) / class_samples.sum()).sum()
            f.write(row + f',{weighted_f1:.4f}\n')

        # Normalize each row of the confusion matrix because class imbalance is
        # high and visualization is difficult
        row_mins = self.confusion_matrix.min(axis=1)
        row_maxs = self.confusion_matrix.max(axis=1)
        cm_norm = (self.confusion_matrix - row_mins[:, None]) / (row_maxs[:, None] - row_mins[:, None])

        # Export Confusion Matrix

        # Replace invalid values with 0
        self.confusion_matrix = np.nan_to_num(self.confusion_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        sns.heatmap(self.confusion_matrix, annot=False, ax=ax, cmap="Blues", fmt="g")

        # Labels, title and ticks
        label_font = {'size': '18'}
        ax.set_xlabel('Predicted labels', fontdict=label_font, labelpad=10)
        ax.set_ylabel('Observed labels', fontdict=label_font, labelpad=10)

        ax.set_xticks(list(np.arange(0.5, len(self.linear_encoder.keys()) - 1 + 0.5)))
        ax.set_yticks(list(np.arange(0.5, len(self.linear_encoder.keys()) - 1 + 0.5)))

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        ax.set_xticklabels([f'{self.crop_encoding[k]} ({k})' for k in sorted(self.linear_encoder.keys()) if k != 0], fontsize=8, rotation='vertical')
        ax.set_yticklabels([f'{self.crop_encoding[k]} ({k})' for k in sorted(self.linear_encoder.keys()) if k != 0], fontsize=8, rotation='horizontal')

        ax.tick_params(axis='both', which='major')

        title_font = {'size': '21'}
        ax.set_title('Confusion Matrix', fontdict=title_font)

        for i in range(len(self.linear_encoder.keys()) - 1):
            ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))

        plt.savefig(self.run_path / f'confusion_matrix_epoch{self.checkpoint_epoch}.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)

        np.save(self.run_path / f'cm_epoch{self.checkpoint_epoch}.npy', self.confusion_matrix)


        # Export normalized Confusion Matrix

        # Replace invalid values with 0
        cm_norm = np.nan_to_num(cm_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        sns.heatmap(cm_norm, annot=False, ax=ax, cmap="Blues", fmt="g")

        # Labels, title and ticks
        label_font = {'size': '18'}
        ax.set_xlabel('Predicted labels', fontdict=label_font, labelpad=10)
        ax.set_ylabel('Observed labels', fontdict=label_font, labelpad=10)

        ax.set_xticks(list(np.arange(0.5, len(self.linear_encoder.keys()) - 1 + 0.5)))
        ax.set_yticks(list(np.arange(0.5, len(self.linear_encoder.keys()) - 1 + 0.5)))

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        ax.set_xticklabels([f'{self.crop_encoding[k]} ({k})' for k in sorted(self.linear_encoder.keys()) if k != 0], fontsize=8, rotation='vertical')
        ax.set_yticklabels([f'{self.crop_encoding[k]} ({k})' for k in sorted(self.linear_encoder.keys()) if k != 0], fontsize=8, rotation='horizontal')

        ax.tick_params(axis='both', which='major')

        title_font = {'size': '21'}
        ax.set_title('Confusion Matrix', fontdict=title_font)

        for i in range(len(self.linear_encoder.keys()) - 1):
            ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))

        plt.savefig(self.run_path / f'confusion_matrix_norm_epoch{self.checkpoint_epoch}.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)

        np.save(self.run_path / f'cm_norm_epoch{self.checkpoint_epoch}.npy', self.confusion_matrix)
        pickle.dump(self.linear_encoder, open(self.run_path / f'linear_encoder_epoch{self.checkpoint_epoch}.pkl', 'wb'))
    def _update_confusion_matrix(self, label, pred):
        num_classes = self.confusion_matrix.size(0)
        indices = label * num_classes + pred
        values = torch.ones_like(label, dtype=torch.float32)
        self.confusion_matrix.view(-1).index_add_(0, indices, values)

class TSViT_lookup(nn.Module):
    """
    Temporal-Spatial ViT5 (used in main results, section 4.3)
    This is a general implementation of lookup position encodings for all dates found in a training set.
    During inference, position encodings are calculated via linear interpolation between dates found in the training data.
    """
    def __init__(self, model_config, train_dates):
        super().__init__()
        train_dates = sorted(train_dates)
        self.train_dates = torch.nn.Parameter(data=torch.tensor(train_dates), requires_grad=False)#.cuda()
        self.eval_dates = torch.nn.Parameter(data=torch.arange(1, 366), requires_grad=False)
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = self.num_patches_1d ** 2
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2  # -1 is set to exclude time feature
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),)
        self.temporal_pos_embedding = nn.Parameter(torch.randn(len(self.train_dates), self.dim), requires_grad=True)
        self.update_inference_temporal_position_embeddings()
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2))

    def forward(self, x, inference=False):
        B, T, C, H, W = x.shape
        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * 365.0001).to(torch.int64)

        if inference:
            self.update_inference_temporal_position_embeddings()
            temporal_pos_embedding = self.get_inference_temporal_position_embeddings(xt).to(x.device)
        else:
            temporal_pos_embedding = self.get_temporal_position_embeddings(xt)

        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(
            B*self.num_classes, self.num_patches_1d**2, self.dim)
        x += self.space_pos_embedding#[:, :, :(n + 1)]
        x = self.dropout(x)
        x = self.space_transformer(x)
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2, self.patch_size**2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return x

    def update_inference_temporal_position_embeddings(self):
        train_dates_idx = torch.arange(self.train_dates.shape[0])
        min_val = torch.min(self.train_dates).item()
        min_idx = torch.argmin(self.train_dates).item()
        if min_val == 0:
            min_val = torch.min(self.train_dates[1:]).item()
            min_idx += 1
        max_val = torch.max(self.train_dates).item()
        max_idx = torch.argmax(self.train_dates).item()
        pos_eval = torch.zeros(self.eval_dates.shape[0], self.dim)
        for i, evdate in enumerate(self.eval_dates):
            if evdate < min_val:
                pos_eval[i] = self.temporal_pos_embedding[min_idx]
            elif evdate > max_val:
                pos_eval[i] = self.temporal_pos_embedding[max_idx]
            else:
                dist = evdate - self.train_dates
                if 0 in dist:
                    pos_eval[i] = self.temporal_pos_embedding[dist == 0]
                    continue
                lower_idx = train_dates_idx[dist >= 0].max().item()
                upper_idx = train_dates_idx[dist <= 0].min().item()
                lower_date = self.train_dates[lower_idx].item()
                upper_date = self.train_dates[upper_idx].item()
                pos_eval[i] = (upper_date - evdate) / (upper_date - lower_date) * self.temporal_pos_embedding[
                    lower_idx] + \
                              (evdate - lower_date) / (upper_date - lower_date) * self.temporal_pos_embedding[upper_idx]
        self.inference_temporal_pos_embedding = nn.Parameter(pos_eval, requires_grad=False)

    def get_temporal_position_embeddings(self, x):
        B, T = x.shape
        index = torch.bucketize(x.ravel(), self.train_dates)
        return self.temporal_pos_embedding[index].reshape((B, T, self.dim))

    def get_inference_temporal_position_embeddings(self, x):
        B, T = x.shape
        index = torch.bucketize(x.ravel(), self.eval_dates)
        return self.inference_temporal_pos_embedding[index].reshape((B, T, self.dim))



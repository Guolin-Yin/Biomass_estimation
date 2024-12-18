import os
import torch
import shutil
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from collections import deque
import json
from pathlib import Path
from tqdm import tqdm
from .train_utils import compute_metrics
class Trainer:
	def __init__(
		self,
		model,
		optimizer,
		loss_fn,
		train_loader = None,
		val_loader = None,
		test_loader = None,
		device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
		prefix=None,
		checkpoint_dir=None,
		max_saved_models=3,
		is_train=True,
		is_transfer_learning=False,
		**kwargs
	):
		"""
		Initializes the Trainer.

		Args:
			model (torch.nn.Module): The regression model to train.
			optimizer (torch.optim.Optimizer): The optimizer.
			loss_fn (torch.nn.Module): The loss function.
			train_loader (torch.utils.data.DataLoader): DataLoader for training data.
			val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
			test_loader (torch.utils.data.DataLoader): DataLoader for test data.
			device (torch.device): Device to run the training on.
			prefix (str): Prefix for checkpoint directory.
			checkpoint_dir (str, optional): Path to checkpoint directory. If None, a default path is created.
			max_saved_models (int): Number of top models to save.
	"""
		self.is_train = is_train
		self.model = model.to(device)
		self.optimizer = optimizer
		self.loss_fn = loss_fn
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_loader = test_loader
		self.device = device
		self.prefix = prefix
		self.is_transfer_learning = is_transfer_learning
		self.max_saved_models = max_saved_models
		self.stage = 'training' if self.is_train else 'testing'
		self.top_models = []
		self.args = kwargs.get('args', None)
		date_time = datetime.now().strftime("%Y-%m-%d->%H:%M:%S")
		if checkpoint_dir:
			self.checkpoint_dir = Path(checkpoint_dir)
			info = f'Test mode in dir: {checkpoint_dir}'
			assert self.checkpoint_dir.exists(), f"Checkpoint directory {checkpoint_dir} does not exist."
			if self.is_transfer_learning and self.is_train:
				self.checkpoint_dir = Path('check_points') / 'transfer_learning' / f'run_{date_time}'
				self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
				self.start_epoch = 1
				self._save_settings()
				info = f"Created new checkpoint directory for transfer learning: {self.checkpoint_dir}"
			self.logger = self._setup_logger(stage=self.stage)
			self.logger.info(info)
			self._load_checkpoint()
		else:
			assert self.is_train, "Cannot create a checkpoint directory for a test model. you should provide a checkpoint_dir"
			self.checkpoint_dir = Path('check_points') / self.prefix / f'run_{date_time}'
			self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
			self.logger = self._setup_logger(stage=self.stage)	
			self.logger.info(f"Created checkpoint directory: {self.checkpoint_dir}")
			self.start_epoch = 1
			self._save_settings()

		# Check for callbacks in kwargs
		self.callbacks = kwargs.get('callbacks', [])
		if self.callbacks:
			self.logger.info(f"Using {len(self.callbacks)} callbacks: {[type(cb).__name__ for cb in self.callbacks]}")
		
		# Check for early stopping callback
		for callback in self.callbacks:
			if type(callback).__name__ == 'EarlyStopping':
				self.early_stop = callback
				self.logger.info(f"Using early stopping with patience {callback.patience}")
				break

		if self.args:
			self.logger.info(f"Using args: {self.args}")
	def _setup_logger(self, stage='training'):
		logger = logging.getLogger(self.prefix)
		logger.setLevel(logging.INFO)

		log_file = self.checkpoint_dir / f'{stage}.log'
		fh = logging.FileHandler(log_file)
		fh.setLevel(logging.INFO)

		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		fh.setFormatter(formatter)

		if not logger.handlers:
			logger.addHandler(fh)

		return logger

	def _save_settings(self):
		settings = {}
		settings['train_params'] = {
			# 'model': str(self.model),
			'optimizer': {k: v for k, v in self.optimizer.param_groups[0].items() if k != 'params'},
			'loss_fn': str(self.loss_fn),
			'device': str(self.device),
			'prefix': self.prefix,
			'checkpoint_dir': str(self.checkpoint_dir),
			'max_saved_models': self.max_saved_models
		}
		try:
			settings['model_params'] = self.model.model_params
		except:
			settings['model_params'] = None
		if self.args:
			settings['args'] = vars(self.args)

		settings_path = self.checkpoint_dir /'settings.json'
		with open(settings_path, 'w') as f:
			json.dump(settings, f, indent=4)

	def _save_checkpoint(self, epoch, val_loss, r2):
		checkpoint_path = self.checkpoint_dir / f'epoch={epoch}_loss={val_loss:.4f}_r2={r2:.4f}.pt'
		checkpoint = {
			'epoch': epoch,
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'val_loss': val_loss,
			'r2': r2
		}
		torch.save(checkpoint, checkpoint_path)

		# Maintain top 3 models with the lowest validation loss
		# self.top_models.append((checkpoint_path, val_loss))
		# self.top_models = sorted(self.top_models, key=lambda x: x[1])

		# Maintain top models with the highest R²
		self.top_models.append((checkpoint_path, r2))
		# Sort in descending order (higher R² is better)
		self.top_models = sorted(self.top_models, key=lambda x: x[1], reverse=True)
		# Remove models that are no longer in the top 3
		while len(self.top_models) > self.max_saved_models:
			# model_to_remove = self.top_models.pop(3)[0]
			model_to_remove = self.top_models.pop(-1)[0]  # Remove the last (worst) model
			if model_to_remove.exists():
				model_to_remove.unlink()  # Use pathlib to remove the file

	def _load_checkpoint(self):
		latest_checkpoint = self._get_latest_checkpoint()
		if latest_checkpoint:
			checkpoint = torch.load(latest_checkpoint, map_location=self.device)
			self.model.load_state_dict(checkpoint['model_state_dict'])
			if self.is_train:
				self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
				self.start_epoch = checkpoint['epoch'] + 1
				self.logger.info(f"Resumed training from epoch {self.start_epoch}")
			else:
				self.logger.info(f"Testing from epoch {checkpoint['epoch']}")

	def _get_latest_checkpoint(self):
		checkpoints = list(self.checkpoint_dir.glob('epoch=*.pt'))
		if not checkpoints:
			return None
		checkpoints.sort(key=lambda x: int(x.stem.split('=')[1].split('_')[0]), reverse=True)
		return checkpoints[2]
	def compute_loss(self, x, y_pred, y_true):
		if y_pred.__class__.__name__ == 'Tensor':
			y_pred = y_pred.squeeze()
			y_true = y_true.squeeze()
		else:
			y_pred = y_pred.output
		# Apply mask to both predictions and ground truth
		valid_mask_y = (y_true > 0)
		valid_mask_x = (x > 0).all(dim=1)
		valid_mask = valid_mask_y & valid_mask_x
  

		valid_predictions = y_pred[valid_mask]
		valid_targets = y_true[valid_mask]

		return self.loss_fn(valid_predictions, valid_targets), valid_mask, y_pred
	def batch_forward_backward(self, 
							data_loader, 
							is_training=True, 
							pbar=None, 
							):

		if is_training:
			self.model.train()  # Set model to training mode
		else:
			self.model.eval()   # Set model to evaluation mode

		total_loss    = 0

		# Initialize accumulators
		# total_samples = 0
		self.predictions, self.actuals, self.image_predictions, self.image_actuals = [], [], [], []
		# Set no_grad context if it's validation (no backpropagation needed)
		context_manager = torch.enable_grad() if is_training else torch.no_grad()

		with context_manager:
			for i, (x, y) in enumerate(data_loader):
				# Check if either x or y is empty
				if x.numel() == 0 or y.numel() == 0:
					continue
				x, y = x.to(self.device), y.to(self.device)
				if is_training:
					self.optimizer.zero_grad()
				# Forward pass

				loss, valid_mask, y_pred = self.compute_loss(x, 
                                         self.model(x), 
                                         y)
				if is_training:
					loss.backward()
					self.optimizer.step()

				# Accumulate total loss
				total_loss += loss.item()
				batch_idx = i + 1
				if pbar:
					pbar.set_postfix(step = batch_idx / len(data_loader), loss=total_loss / batch_idx)
					pbar.update(1)
				
				# Get valid targets and predictions
				valid_targets = y[valid_mask]
				valid_predictions = y_pred[valid_mask]
				# Store pixel-level predictions
				self.actuals.extend(valid_targets.clone().detach().cpu().numpy().flatten())
				self.predictions.extend(valid_predictions.clone().detach().cpu().numpy().flatten())
				# Store image-level predictions (mean per image in batch)
				if not (type(self.model).__name__ == 'TFCNN'):
					for img_idx in range(y.size(0)):
						img_mask = valid_mask[img_idx]
						if img_mask.any():
							img_pred_mean = valid_predictions.mean().item()
							img_actual_mean = valid_targets.mean().item()
							self.image_predictions.append(img_pred_mean)
							self.image_actuals.append(img_actual_mean)
		# Check for early stopping
		# Compute average loss
		avg_loss = total_loss / len(data_loader)
		r2, mae, rmse = compute_metrics(self.actuals, self.predictions)
		if not (type(self.model).__name__ == 'TFCNN'):
			r2_image, mae_image, rmse_image = compute_metrics(self.image_actuals, self.image_predictions)
			return avg_loss, r2_image, mae_image, rmse_image
		else:
			return avg_loss, r2, mae, rmse

	def train(self, num_epochs, run_test_at_the_end=False):
		assert self.is_train, "Cannot train a test model."
		for epoch in range(self.start_epoch, num_epochs+1):
			# Training phase
			with tqdm(total=len(self.train_loader), desc=f"Training [{epoch}/{num_epochs}]", leave=False) as pbar:
				train_loss, train_r2, train_mae, train_rmse = self.batch_forward_backward(self.train_loader, is_training = True, pbar = pbar)
			
   			# Validation phase
			with tqdm(total=len(self.val_loader), desc="Validation", leave=False) as pbar:
				val_loss, val_r2, val_mae, val_rmse = self.batch_forward_backward(self.val_loader, is_training=False, pbar = pbar)
      			# Save checkpoint
			self._save_checkpoint(epoch, val_loss, val_r2)
			self.logger.info(f"Epoch [{epoch}/{num_epochs}] completed." \
                            f"Avg Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, " \
                            f"Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}, " \
                            f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}, " \
                            f"Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
			# Check for early stopping
			if hasattr(self, 'early_stop') and self.early_stop is not None:
				if self.early_stop(val_loss):
					self.logger.info(f"Early stopping triggered at epoch {epoch}.")
					break
		if run_test_at_the_end:
			self.is_train = False
			self.logger.info("Running test at the end of training.")
			self._load_checkpoint()
			self.test()
	def test(self):
		"""
		Tests the model and saves the results to the checkpoint path.
		"""
		assert not self.is_train, "Cannot test a train model."
		# Calculate metrics
		with tqdm(total=len(self.test_loader), desc="Testing", leave=False) as pbar:
			_ = self.batch_forward_backward(self.test_loader, is_training=False, pbar = pbar)

		# Save results to CSV
		results_path = self.checkpoint_dir / 'test_results.csv'
		df_pixels = pd.DataFrame({
			'Actual': self.actuals,
			'Predicted': self.predictions
		})
		df_pixels.to_csv(results_path, index=False)
		r2, mae, rmse = compute_metrics(self.actuals, self.predictions)
		# Log metrics
		self.logger.info(f"Test Results - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
		self.logger.info(f"Test results saved to {results_path}")

		if not (type(self.model).__name__ == 'TFCNN'):
			results_path_image = self.checkpoint_dir / 'test_results_image.csv'
			df_image = pd.DataFrame({
				'Actual': self.image_actuals,
				'Predicted': self.image_predictions
			})
			df_image.to_csv(results_path_image, index=False)
			r2_image, mae_image, rmse_image = compute_metrics(self.image_actuals, self.image_predictions)
			self.logger.info(f"Test Results - R²: {r2_image:.4f}, RMSE: {rmse_image:.4f}, MAE: {mae_image:.4f}")
			self.logger.info(f"Test results saved to {results_path_image}")

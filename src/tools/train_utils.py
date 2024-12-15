from sklearn.metrics import r2_score as sklearn_r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
import json
from pathlib import Path
def load_checkpoint_cfg(checkpoint_dir):
    with open(Path(checkpoint_dir) / 'settings.json', 'r') as f:
        settings = json.load(f)
        # load model_params in model_params key
        model_params = settings['model_params']
        # load train_params in train_params key
        train_params = settings['train_params']
        return model_params, train_params
    
def compute_metrics(y_true, y_pred):
    r2 = sklearn_r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return r2, mae, rmse
# def compute_metrics(predictions, actuals):
#     """
#     Compute R² at both pixel and image levels, along with MAE and RMSE
#     """
#     predictions = np.array(predictions)
#     actuals = np.array(actuals)
    
#     # Pixel-level metrics (over all valid pixels)
#     pixel_r2 = r2_score(actuals, predictions)
#     mae = mean_absolute_error(actuals, predictions)
#     rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
#     return pixel_r2, mae, rmse
class EarlyStopping:
	def __init__(self, patience=20, min_delta=0):
		"""
		Initializes the early stopping instance.

		Parameters:
		- patience (int): Number of epochs to wait for improvement before stopping.
		- min_delta (float): Minimum change in validation loss to qualify as an improvement.
		"""
		self.patience = patience
		self.min_delta = min_delta
		self.best_loss = float('inf')
		self.epochs_without_improvement = 0
		self.should_stop = False

	def __call__(self, current_loss):
		"""
		Checks if training should stop based on the current validation loss.

		Parameters:
		- current_loss (float): The validation loss of the current epoch.

		Returns:
		- bool: True if early stopping is triggered, False otherwise.
		"""
		if current_loss < self.best_loss - self.min_delta:
			self.best_loss = current_loss
			self.epochs_without_improvement = 0
		else:
			self.epochs_without_improvement += 1

		if self.epochs_without_improvement >= self.patience:
			self.should_stop = True

		return self.should_stop
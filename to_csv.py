import csv
from collections import deque

class LossLogger:
    def __init__(self, file_name='loss.csv'):
        self.file_name = file_name
        self.train_losses = deque()
        self.val_losses = deque()
        # Check if file exists, if not, write headers
        try:
            with open(self.file_name, 'x', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Train loss', 'Val loss'])
        except FileExistsError:
            pass

    def log_train_loss(self, loss):
        self.train_losses.append(loss)
        self.write_losses()

    def log_val_loss(self, loss):
        self.val_losses.append(loss)
        self.write_losses()

    def write_losses(self):
        with open(self.file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            while self.train_losses and self.val_losses:
                writer.writerow([self.train_losses.popleft(), self.val_losses.popleft()])
            # If there are unmatched train losses, write them with an empty value for val loss
            while self.train_losses:
                writer.writerow([self.train_losses.popleft(), ''])
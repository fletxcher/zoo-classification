
import os
import json 
from tqdm import tqdm
from datetime import datetime
import pickle
import numpy as np
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import io, base64
from IPython.display import HTML

class ZCMLP(MLPClassifier):
    def __init__(self, activation: str, batch_size: int, epochs: int, hidden_layer_sizes: tuple, solver: str, tol: float, lr: float, verbose: bool, random_state: int):
        super().__init__(activation = activation, batch_size = batch_size, max_iter = epochs, hidden_layer_sizes = hidden_layer_sizes, solver = solver, tol = tol, learning_rate_init = lr, learning_rate = 'constant', verbose = verbose, random_state = random_state)
        self._train_loss = []
        self._val_loss = []
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation = activation
        self.hidden_layer_sizes = hidden_layer_sizes
        self.solver = solver
        self.tol = tol
        self.lr = lr
        self.random_state = random_state

    def architecture(self):
        print(json.dumps(self.get_params(), indent = 4))

    def train(self, features: pd.DataFrame, targets: pd.Series, train_size: float, test_size: float):
        self.features = features
        self.targets = targets
        self.classes = sorted(targets.unique().tolist())
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(features, targets, train_size = train_size, test_size = test_size, random_state = 32)
        self.x_test, self.x_val, self.y_test, self.y_val = train_test_split(self.x_test, self.y_test, train_size = train_size, test_size = test_size, random_state = 5)
        pbar = tqdm(range(self.epochs), desc = "Training", leave = True)
        for epoch in pbar:
            for b in range(self.batch_size, len(self.y_train), self.batch_size):
                x_batch, y_batch = self.x_train[b - self.batch_size:b], self.y_train[b - self.batch_size:b]
                self.partial_fit(x_batch, y_batch, classes=self.classes)
                self._train_loss.append(self.loss_)
                val_loss = log_loss(self.y_val, self.predict_proba(self.x_val), labels = self.classes)
                self._val_loss.append(val_loss)

            pbar.set_postfix({
                "Epoch": f"{epoch}/{self.epochs}",
                "Train Loss": f"{self._train_loss[-1]:.4f}",
                "Val Loss": f"{val_loss:.4f}"
            })
        return self
    
    def visualizations(self):
        matplotlib.rcParams['animation.embed_limit'] = 50000000
        fig, ax = plt.subplots(figsize = (24, 8))
        ax.set_title('Training Vs Validation Loss')
        _train, = ax.plot([], [], 'b-', label = 'Training Loss')
        _val, = ax.plot([], [], 'r:', label = 'Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.grid(visible = True)
        ax.legend()

        x = [i for i in range(0, self.epochs + 1)]
        y1 = self._train_loss
        y2 = self._val_loss    

        ax.set_xlim(0, self.epochs)
        ax.set_ylim(min(min(y1), min(y2)), max(max(y1), max(y2)))

        def init():
            _train.set_data([], [])
            _val.set_data([], [])
            return _train, _val

        def create_animation(frame):
            _train.set_data(x[:frame + 1], y1[:frame + 1])
            _val.set_data(x[:frame + 1], y2[:frame + 1])
            return _train, _val

        animation = FuncAnimation(fig = fig, func = create_animation, frames = self.epochs, init_func = init, interval = 30, blit = True)
        save_dir = f'{cwd}/animations/{datetime.now().strftime("%m-%d-%Y ~ %H:%M:%S")}'
        os.makedirs(save_dir, exist_ok = True)
        animation.save(f'{save_dir}/animation.mp4', writer = 'ffmpeg', fps = 60)

    def evaluation(self):
        y_predictions = self.predict(self.x_test)
        results = pd.DataFrame(self.y_test)
        results['predictions'] = y_predictions
        self.accuracy = round(number = (accuracy_score(self.y_test, y_predictions) * 100), ndigits = 2)
        self.f1 = round(number = (f1_score(y_true = self.y_test, y_pred = y_predictions, average = 'macro') * 100), ndigits = 2)

        fig, ax = plt.subplots(figsize=(10, 10)) 
        ax.set_title("Confusion Matrix")
        classes = self.targets.unique().sort()
        ConfusionMatrixDisplay.from_predictions(
            y_true = self.y_test,
            y_pred = y_predictions,
            display_labels = classes,
            ax = ax
        )

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        html = f"""
        <div style="display: flex; justify-content: space-around; align-items: flex-start;">
            <div>
                <div>{f'Accuracy: {self.accuracy} %'}</div>
                <div>{f'F1 Score: {self.f1} %'}</div>
            </div>
            <div>{results.to_html()}</div>
            <div><img src="data:image/png;base64,{b64}"/></div>
        </div>
        """
        save_dir = f'{cwd}/evaluations/{datetime.now().strftime("%m-%d-%Y ~ %H:%M:%S")}'
        os.makedirs(save_dir, exist_ok = True)
        with open(f'{save_dir}/evaluation.html', 'w') as file:
            file.write(html)

        print(f'Accuracy: {self.accuracy} %')
        print(f'F1 Score: {self.f1} %')

    def save(self, cwd : str):
        save_dir = f'{cwd}/models/{datetime.now().strftime("%m-%d-%Y ~ %H:%M:%S")}'
        os.makedirs(save_dir, exist_ok = True)
        with open(f'{save_dir}/zcmlp.pkl', 'wb') as file:
            pickle.dump(self, file)
        file.close()
        with open(f'{save_dir}/metrics.txt', 'w') as file:
            file.write(f'\nTraining Cycle Metrics:\nNumber Of Epochs: {self.epochs}\nAverage Training Loss: {np.mean(self._train_loss):.2f}\nAverage Validation Loss: {np.mean(self._val_loss):.2f}\nAccuracy: {self.accuracy} %\nF1 Score: {self.f1} %\n')
        file.close()

# Load The Dataset
cwd = os.getcwd()
df1 = pd.read_csv(filepath_or_buffer = f'{cwd}/data/zoo1.csv')
df2 = pd.read_csv(filepath_or_buffer = f'{cwd}/data/zoo2.csv')
df3 = pd.read_csv(filepath_or_buffer = f'{cwd}/data/zoo3.csv')
df = pd.concat([df1, df2, df3], ignore_index = True)
df = df.drop(columns = ['animal_name'])

# Extract The Features And Targets
features = df.drop(columns = ['class_type'])
targets = df['class_type']

# Define An MLPClassifier & Train
epochs = 1000
batch_size = 10
lr = 1e-3
model = ZCMLP(activation = 'relu', batch_size = batch_size, epochs = epochs, hidden_layer_sizes = (len(features.columns),), solver = 'sgd', tol = 1e-4, lr = 1e-3, verbose = False, random_state = 1)
model.architecture()
model.train(features = features, targets = targets, train_size = 0.7, test_size = 0.3)

# Visualize The Training Process
model.visualizations()

# Make Predictions Based On Training Data And Evaluate The Model
model.evaluation()

# Save The Model Using Pickle
model.save(cwd)
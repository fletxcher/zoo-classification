
# Zoo Animal Classification

This project is a Jupyter Notebook for classifying zoo animals using a custom Multi-Layer Perceptron (MLP) implementation. It leverages datasets from Kaggle to train and evaluate a neural network model for multi-class classification.

## Datasets

- [UCI Zoo Animal Classification Dataset](https://www.kaggle.com/datasets/uciml/zoo-animal-classification)
- [Zoo Animals Extended Dataset](https://www.kaggle.com/datasets/agajorte/zoo-animals-extended-dataset)

## Project Structure
```
├── README.md
├── data
│   ├── zoo1.csv
│   ├── zoo2.csv
│   └── zoo3.csv
├── requirements.txt
└── zcmlp.py
```

## Features

- Loads and merges multiple zoo animal datasets.
- Extracts features and targets for classification.
- Defines and trains a custom MLP classifier ([`ZCMLP`](zcmlp.py)).
- Visualizes training progress and model architecture.
- Evaluates model performance and displays metrics.
- Saves trained models and metrics for future use.

## Usage

1. **Install dependencies**  
pip install -r requirements.txt

2. **Download datasets**  
Place the CSV files from the Kaggle datasets above into the [`data`](data) directory as `zoo1.csv`, `zoo2.csv`, and `zoo3.csv`.

3. **Run the script**  
Open [`zcmlp.py`](zcmlp.py) in VS Code and execute the program.
4. **Model Output**  
- Training visualizations and evaluation metrics are displayed in the notebook.
- Trained models and metrics are saved in the [`models`](models) directory.

## Model

The notebook uses a custom neural network class [`ZCMLP`](zcmlp.py) for training and evaluation. You can modify hyperparameters such as epochs, batch size, and learning rate in the notebook.

## References

- [UCI Zoo Animal Classification Dataset](https://www.kaggle.com/datasets/uciml/zoo-animal-classification)
- [Zoo Animals Extended Dataset](https://www.kaggle.com/datasets/agajorte/zoo-animals-extended-dataset)

---

For more details, see [`zcmlp.py`](zcmlp.py).
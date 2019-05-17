# Similar Bonds Recommender

## Code Layout

The code in this repository is laid out as follows:

- `data.py` - Contains code for loading raw data
- `model.py` - Contains code for training the distance-based model

For the feedback-based model, see the following files:

- `BPRDataPreparation.ipynb` - Contains code that demonstrates how to use the distance-based models to bootstrap the feedback-based model
- `BPRModel.ipynb` - Contains code that defines and trains the Bayesian Personalized Recommendation model, and demonstrates how to incorporate user feedback in an online setting

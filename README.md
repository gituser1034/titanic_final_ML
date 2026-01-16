## Machine Learning Final - Titanic Guessing Game

**Authors:** Olly Love, Nathan Singer, David Kelly

This is a comparative study of different Machine Learning approaches.


## What we did:

We have investigated various neural network design choices and compared them using the [Kaggle Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset).

##
**We have compared**:
- Linear Classification
- Logistic Regression
- MLP
    - 1D
    - 2D
    - 4D with a Residual Network

We also put further research into expanding the model usage, by implementing a Tabular Autoencoder.


## Conclusions

Not every design applies to every use case.

Some of the designs we chose, Linear, Logistic, MLP, showed great results in classifying our data, albeit in different capacities.

Some we chose, in particular the Tabular Autoencoder, simply do not work for this type of dataset.

After a thourough investigation, we have determined that MLP 4D outputs the best results for classifying the Titanic data, and have deployed it into a simple guessing game.

## How does the game work?

The user is taken through a series of prompts. Each relays information about a theoretical, or real, passenger that was aboard the Titanic. With the information they are given, the user is expected to determine if the passenger survived the disaster or not.

The answer is retrieved using the model we have trained on the Titanic dataset.

## Installing and running

To run, clone this repository.

Run `pip install -r requirements.txt` to install all of the required packages.

Run every cell in main.ipynb to walk through our research, train, and store a copy of each model.
**Alternatively**, use the pretrained models provided in the `models/` folder.

Then, play Titanic Guesser with `python deployment_game.py` to see the MLP model in action.





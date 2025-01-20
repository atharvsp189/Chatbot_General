# Chatbot using TensorFlow

This repository contains the code for a chatbot built using TensorFlow. The chatbot is trained on a dataset of intents and patterns, and is able to respond to user queries in a natural language way.

## Dependencies

This project requires the following Python libraries:

* random
* json
* pickle
* numpy
* tensorflow
* nltk

## Files

* `chatbot.py`: This file contains the core logic of the chatbot, including loading the data, training the model, and making predictions.
* `words.pkl`: This file stores a pickled list of words used in the training data.
* `classes.pkl`: This file stores a pickled list of the chatbot's intent classes.
* `intents.json`: This file defines the chatbot's intents and patterns.
* `chatbot_model.h5`: This file stores the trained TensorFlow model.

## Usage

To run the chatbot, you will need to have the above dependencies installed. Once you have installed the dependencies, you can run the chatbot by following these steps:

1. Clone this repository.
2. Open a terminal in the project directory.
3. Run the following command to install the dependencies:

```bash
pip install -r requirements.txt

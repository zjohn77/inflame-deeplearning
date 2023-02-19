## Sentiment Classifier Training App
### Install
Go inside the sentiment_imdb directory and then execute the following:

`python -m venv venv`

`source venv/bin/activate`

`pip install -r requirements.txt`

### Run
Run the trainer module from its entry point as follows:

`python -m trainer.task`

The above should write to the configured cloud bucket the sentiment classification of a sample review.
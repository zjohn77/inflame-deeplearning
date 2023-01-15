from sentiment_classifier import *
import torch
from sentiment_classifier.config import CLASS_VOCAB


def main():
    # Train model and then save it to model_save_path.
    xtrain, ytrain, input_features, num_classes = process_data()
    model_save_path = train(xtrain, ytrain, input_features, num_classes)

    # Load the locally saved model and use it to generate predictions on new xtrain rows.
    iris_classifier = torch.jit.load(model_save_path)
    predictions = predict_class(xtrain[0:5], iris_classifier)
    print([CLASS_VOCAB[class_index] for class_index in predictions])


if __name__ == "__main__":
    main()

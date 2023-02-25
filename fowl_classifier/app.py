import torch
from fowl_classifier import cli_main, load_model, run_inference

if __name__ == "__main__":
    INPUT_DATA_FOLDER_NAME = "fowl_data"
    cli_main()

    prediction = run_inference(load_model())
    output_path = os.path.join(ModelDirStructure().inference_output, "prediction.json")
    try:
        with open(output_path, "w") as f:
            json.dump(prediction, f)
    except FileNotFoundError:
        os.mkdir(ModelDirStructure().inference_output)
        with open(output_path, "w") as f:
            json.dump(prediction, f)

# Automated HVAC Fault Detection with Data-Driven Models and Feature Engineering Using Luigi Pipelines
 This study presents a systematic and automated method for detecting HVAC faults using the Luigi pipeline framework. The approach is built around two modular pipelines. The first is a preprocessing pipeline, which manages tasks such as data cleaning, transformation, and feature engineering. The second is a classification pipeline, designed to handle model training, validation, and evaluation.

 ## System requirements
* Linux or macOS or Windows
* Python 3.11.5
* All the dependencies are listed in requirements.txt file

## Instructions

###  Luigi
All the datasets, pre-processed dataset and configuration files for both preprocessing pipeline and classification pipeline are given here:  [google drive link](https://drive.google.com/drive/folders/1qmR-28G64zJOxQRX7RNLIpBw2aJX226f?usp=sharing). The Nine Feature Engineering Configuraiton are given seperately as yaml file. The configuration with ARIMA, LSTM, Random Forest and Fully Connected Neural Network is also provided here.
Luigi is an open-source Python framework designed for building and managing complex pipelines for data processing, workflow automation, and task scheduling. To learn more about Luigi, please refer to the official documentation: [Luigi Documentation](https://luigi.readthedocs.io/en/stable/).

To initiate the Luigi scheduler, execute the following command in your terminal:

```bash
luigid
```

### Pre-processing Pipeline
For the preprocessing pipeline, two main directories need to be created: `input_dir` and `output_dir`. Within the `input_dir` directory, create an additional subdirectory named `input`. Place the dataset that needs to be preprocessed inside this `input` directory.

The required directory structure is as follows:  
├── input_dir/  
│ └── input/   
│ └── dataset.csv  
├── output_dir/  
The preprocessing pipeline will process this dataset and generate the results in the `output_dir` directory.
Place the pre-processing configuration file in the project_directory.
To run the pre-processing pipeline execute the command:
```bash
python .\preprocessing_pipeline.py -I {path_to_input_directory} -O {path_to_output_directory} -C {path_to_configuration_file}
eg.
python .\preprocessing_pipeline.py -I input_dir -O output_dir -C preprocessing_config.yaml

```

The preprocessed data will be located within the output directory, organized under a subdirectory named after the final task executed in the preprocessing pipeline. 

### Classification Pipeline
The Classification Pipeline has a similar structure. Two  directories need to be created: `classification_input_dir` and `classification_output_dir`. Within the `classification_input_dir` directory, create an additional subdirectory named `input`. Place the pre-processed dataset inside this `input` directory.

The required directory structure is as follows:
├── classification_input_dir/  
│ └── input/   
│ └── pre_processed_dataset.csv  
├── classification_output_dir/  

The Classification pipeline will perform splitting, training,validation and evaluation on this dataset and generate the results in the `classification_output_dir` directory.
Place the classification configuration file in the project_directory.
To run the classification pipeline execute the command:
```bash
python .\classification_pipeline.py -I {path_to_input_directory} -O {path_to_output_directory} -C {path_to_configuration_file}
eg.
python .\classification_pipeline.py -I classification_input_dir -O classification_output_dir -C classification_config.yaml

```

The evaluation result data will be located within the output directory, organized under a subdirectory named 'evaluation' after all the task has been executed in the classification pipeline. 
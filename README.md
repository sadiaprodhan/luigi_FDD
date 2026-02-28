# Detecting, Diagnosing and Explaining HVAC Faults: A Series of Case Studies
This study presents a systematic and automated approach for HVAC fault detection using the Luigi pipeline framework. The method is organized into four modular pipelines:

- **Data preparation (preprocessing) pipeline:** performs dataset splitting, data cleaning, transformation, and feature engineering.
- **Binary classification (classification) pipeline:** handles model training, validation, and evaluation.
- **Multiclass classification pipeline:** supports training, validation, and evaluation for multiple fault categories.
- **Brick-integrated diagnosis pipeline:** uses SHAP-based explanations and component-level fault localization to support interpretable diagnosis.
## System requirements
* Linux or macOS or Windows
* Python 3.11.5
* All the dependencies are listed in requirements.txt file
## Instructions

###  Luigi
All the datasets, pre-processed dataset, configuration files along with the results of the four pipelines are given [here:](https://drive.google.com/drive/folders/1zUX5VBvjk8Po63HZqxI7NuSJUjTbmRCa?usp=sharing). 

Luigi is an open-source Python framework designed for building and managing complex pipelines for data processing, workflow automation, and task scheduling. To learn more about Luigi, please refer to the official documentation: [Luigi Documentation](https://luigi.readthedocs.io/en/stable/).

To initiate the Luigi scheduler, execute the following command in your terminal:

```bash
luigid
```
### Data preparation (preprocessing) pipeline
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
### Binary Classification Pipeline
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
### Multiclass Classification Pipeline
The Multi Classification Pipeline has a similar structure. Two  directories need to be created: `multi_classification_input_dir` and `multi_classification_output_dir`. Within the `multi_classification_input_dir` directory, create an additional subdirectory named `input`. Place the pre-processed dataset inside this `input` directory.

The required directory structure is as follows:
├── multi_classification_input_dir/  
│ └── input/   
│ └── pre_processed_dataset.csv  
├── multi_classification_output_dir/  

The Classification pipeline will perform splitting, training,validation and evaluation on this dataset and generate the results in the `multi_classification_output_dir` directory.
Place the classification configuration file in the project_directory.
To run the classification pipeline execute the command:
```bash
python .\multi_classification_pipeline.py -I {path_to_input_directory} -O {path_to_output_directory} -C {path_to_configuration_file}
eg.
python .\multi_classification_pipeline.py -I multi_classification_input_dir -O multi_classification_output_dir -C multi_classification_config.yaml

```
### Brick Integrated Diagnosis Pipeline
The Brick Pipeline has a similar structure. Two  directories need to be created: `brick_input_dir` and `brick_output_dir`. Within the `brick_input_dir` directory, create an additional subdirectory named `input`. Place the .ttl (brick file), all the 5 fold joblibs of the model, all the test sets and the 5 fold predicted file inside this `input` directory (Total 16 input files).

The required directory structure is as follows:
├── brick_input_dir/  
│ └── input/   
│ └── LBNL_FDD_Data_Sets_SDAHU_ttl.ttl<br>
│ └── predicted_fold_1.csv (5 files)<br>
│ └── randomforest_fold_1.joblib (5 files)<br>
│ └── SDAHU_FULL_M_CLASS_X_test_fold_1.joblib (5 files) <br>
├── brick_output_dir/  

The Brick pipeline performs **signal-to-component mapping**, **SHAP computation**, **top-sensor identification**, and **component-level localization**, and writes all outputs to the `brick_output_dir` directory.

Place the Brick configuration file in the `project_directory`.To run the brick pipeline execute the command:
```bash
python .\brick_pipeline.py -I {path_to_input_directory} -O {path_to_output_directory} -C {path_to_configuration_file}
eg.
python .\brick_pipeline.py -I brick_input_dir -O brick_output_dir -C brick_pipeline_config.yaml

```
To ensure all tasks run correctly, each task is assigned an ID, and each subsequent task uses the previous task’s ID as its input ID.
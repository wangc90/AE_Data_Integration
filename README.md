# Auto encoders with shared and specific embeddings for multi-omic data integration
## Required Python libraries to replicate this analysis:
torch<br />
pandas<br />
numpy<br />
sklearn<br />
optuna<br />
seaborn<br />
rpy2.ipython<br />

## Required Python scripts to run the AE_results.ipynb in the Evaluation_Auxiliary folder
download these scripts from Evaluation_Auxiliary folder to access them <br />
Data_prep.py<br />
tsn_visulization.py<br />
reconloss_from_retrained_models.py<br />
embedding_from_retrained_models.py<br />
nb_classification.py<br />
model_structures.py<br />

## Required R library to replicate the figures in the AE_results.ipynb
tidyr<br />
dplyr<br />
ggplot2<br />
ggsci<br />
scales<br />
RColorBrewer<br />
pheatmap<br />
ggpubr<br />

## Required Code from MOCSS paper
download the folder from Simulation_Auxiliary or TCGA_Auxiliary/code: these folders contain the codes from MOCSS paper that are used in this study
point the python path to this folder to access these code<br />
sys.path.insert(1, 'Simulation_Auxiliary')

## How to replicate the results (e.g.,on simulation data):

Within the Simulation_Models folder, model selection and retraining codes are included for 8 different AE architectures. 

For example, to select the optimal hyperparameters for the CNC_AE model with the simulation data, simply run the CNC_AE_model_selection.py (You should install the necessary Python packages before importing them and change the folder path accordingly). The first part of the code defines the CNC_AE model structure with CNC_Encoder and CNC_Decoder as well as the CNC_AE block that uses the CNC_Encoder and CNC_Decoder. Following the model structure, we define the custom loss for the CNC_AE model. We then defined the model training step (Objective_CV) that uses the Objective function defined earlier to minimize the loss function and return the averaged loss on the validation set. We also created the pytorch tensor Dataset that is specific for the simulation data training. Finally, the CNC_AE_model_selection function is called to run 50 Optuna trials to select the optimal hyperparameter sets corresponding to the lowest averaged reconstruction loss. The validation loss and the Optuna results will be saved locally for subsequent use when retraining the model.

Once the optimal hyperparameter set is selected, we can then fix these hyperparameters and retrain the model on the entire training set to obtain the trained model that is eventually used for model evaluation (i.e., calculate the reconstruction loss and extract the embedding for subsequent classification task). To retrain the model, simply run the CNC_AE_retraining.py which automatically loads the saved optimal hyperparameter sets for retraining. This Python script will save the retrained model weights locally for subsequent model evaluation.

Note, to run the above CNC_AE_model_selection.py and CNC_AE_retraining.py, you need to import some of the functions from the Simulation_Auxiliary folder (codes adopted from the MOCSS paper).

To evaluate the model performance, follow the AE_results.ipynb within Evaluation_Auxiliary folder and make sure to import the necessary auxiliary codes from the Evaluation_Auxiliary folder.

## To replicate the results for TCGA data, follow the above step and use the corresponding TCGA dataset and TCGA model folder.

## Organization of this Repository

### The Simulation_Data:
contains the 20 simulation datasets with different levels of difficulties used to compare the 8 AE architectures implemented in this study.

### The Simulation_Models:
contains the implementation of 8 AE architectures used for simulation datasets in this study.

### Simulation_Auxiliary
contains the auxiliary code accompanying the models used for the simulation dataset. 

### The TCGA_Data:
contains the pre-processed TCGA datasets of 6 primary cancer types downloaded from https://acgt.cs.tau.ac.il/multi_omic_benchmark/download.html.

### The TCGA_Models:
contains the implementation of 8 AE architectures used for TCGA datasets in this study.

### TCGA_Auxiliary
contains the auxiliary code accompanying the models used for the TCGA dataset. 

### Evaluation_Auxiliary
contains the auxiliary code accompanying the model evaluation jupyternotebook: AE_results.ipynb

### AE model architectures implemented in this study
![AE models](Images/AE_models.png)

Different AE model structures compared in this study. a): AE with direct X1, X2 concatenated together as the model input, referred as CNC AE;
b): AE with processed X1, X2 concatenated together as the model input, referred as X AE; c): AE with different concatenation of processed X1, X2 as
the model input, referred as MM AE; d): AE structure from previous study with orthogonal constraints imposed during the post-processing, referred
as MOCSS [2]; e): AE with both individual and concatenated X1, X2 as the model input with addition imposed orthogonal constraints, referred as
JISAE-O; f): similarly to model in e) without additional orthogonal constraints, referred as JISAE.



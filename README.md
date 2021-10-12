# Tf-GCZSL
Paper: Task-free Generalized Continual Zero-shot Learning (Tf-GCZSL) 
Link: https://arxiv.org/abs/2103.10741v1

## To run the code for task-agnostic setting follow the following instructions:

1) Go to 'tfgczsl_task_agnostic' folder
2) First run the 'data_loader.py' script to process the data. 
3) Run the 'tfgczsl.py' to get the results in .csv file. (Harmonic Mean, Seen Accuracy, Unseen Accuracy) 

Note: 
=
      * The path variable must be changed to point to the dataset location. (CUB Dataset)

      * The Dataset can be downloaded from https://www.dropbox.com/sh/mrofy57ci4jcmx5/AACeN9zNpAe_ZyNi0_iouASma?dl=0 

      * For Without DER set 'use_der' variable in 'tfgczsl.py' to False 


## To run the code for task_free setting follow the following instructions:

1) Go to the 'tfgczsl_taskfree' folder
2) First run the 'data_loader.py' script to process the data.
3) Run 'TFGCZSL_STG1_MB.py' to get the results for M_b setting in a .csv file.(Harmonic Mean, Seen Accuracy, Unseen Accuracy)
4) Run 'TFGCZSL_STG2_MST.py' to get the results for M_st setting in a .csv file.(Harmonic Mean, Seen Accuracy, Unseen Accuracy)

Note: 

      * The path variable must be changed to point to the dataset location. (CUB Dataset)
      
      * The Dataset can be downloaded from https://www.dropbox.com/sh/mrofy57ci4jcmx5/AACeN9zNpAe_ZyNi0_iouASma?dl=0
      
      * For Without DER set 'use_der' variable in 'TFGCZSL_STG1_MB.py' and 'TFGCZSL_STG2_MST.py' to False 


# IPSOCNV
IPSOCNV is a copy number variation detection method using neural networks combined with improved particle swarms

# Document Description

**Training:** The training set used to train the model is saved<br>
**RealData:** Three real datasets used by the project are saved<br>
**SimulationData:** The simulation dataset used in the project at a purity of 0.2 is saved<br>
**Code:** Saved the complete code of the project<br>
**result:** Saved two example images of the project<br>

# Usage of IPSOCNV
## step1: Data preprocessing
After sorting and alignment of sample files with samtools and BWA, simulation.py performs further preprocessing operations, including feature extraction of other parameters, to obtain simulated data used for experiments. The same processing is performed on real data as well.
## step1: Training process
Importing training data and execute train_ipsocnv.m
## step2: Testing process
You can select the appropriate simulation dataset or the real dataset, then execute bpcnv_test.m and bpcnv_test_real.m respectively
## step3: Analysis of results
The results are output directly as precision and sensitivity, and the test code can also be modified to obtain the location and type of copy number variants detected

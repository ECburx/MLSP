# Machine Learning in Predicting Aqueous Solubility

The goal is to evaluate and compare the effectiveness of some of the most popular machine learning modeling methods and molecular featurization techniques in predicting aqueous solubility. 

## ML Algorithms

**2D descriptor-based**
1. XGBoost
2. LightGBM
3. 1D-CNN
4. TabNet

**Graph-based**
1. GCN
2. GAT
3. GATv2
4. GIN
5. AttentiveFP
6. MGCN
7. MPNN
8. NF
9. PAGTN
10. OGB
11. Weave
12. SPGNN

More will be coming...

## Data

1. 2008 Solubility Challenge Dataset [^1][^2]
3. 2019 Solubility Challenge Dataset [^3][^4]
4. DLS-100 [^6][^7][^8]
5. AqSolDB [^9]
6. AQUA [^10][^11][^12]
7. PHYS [^10]
8. [^13][^14][^15][^16][^17][^18][^19][^20]
9. 1st EUOS/SLAS Joint Solubility Prediction Challenge Dataset [^5]

Featurized dataset can be found in the `/data` directory.

[^1]: Llinàs, A.; Glen, R. C.; Goodman, J. M. Solubility Challenge: Can You Predict Sol-ubilities of 32 Molecules Using a Database of 100 Reliable Measurements? Journal of Chemical Information and Modeling 2008, 48, 1289–1303.
[^2]: Hopfinger, A. J.; Esposito, E. X.; Llinàs, A.; Glen, R. C.; Goodman, J. M. Findings of the Challenge To Predict Aqueous Solubility. Journal of Chemical Information and Modeling 2008, 49, 1–5.
[^3]: Llinas, A.; Avdeef, A. Solubility Challenge Revisited after Ten Years, with Multilab Shake-Flask Data, Using Tight (SD ∼ 0.17 log) and Loose (SD ∼ 0.62 log) Test Sets. Journal of Chemical Information and Modeling 2019, 59, 3036–3040.
[^4]: Llinas, A.; Oprisiu, I.; Avdeef, A. Findings of the Second Challenge to Predict Aqueous Solubility. Journal of Chemical Information and Modeling 2020, 60, 4791–4803.
[^5]: Hunklinger, A.; Hartog, P.; Šícho, M.; Godin, G.; Tetko, I. V. The openOCHEM consensus model is the best-performing open-source predictive model in the First EUOS/SLAS joint compound solubility challenge. SLAS Discovery 2024, 29, 100144.
[^6]: McDonagh, J. L.; Nath, N.; Ferrari, L. D.; van Mourik, T.; Mitchell, J. B. O. Uniting Cheminformatics and Chemical Theory To Predict the Intrinsic Aqueous Solubility of Crystalline Druglike Molecules. Journal of Chemical Information and Modeling 2014, 54, 844–856.
[^7]: Boobier, S.; Osbourn, A.; Mitchell, J. B. O. Can human experts predict solubility better than computers? Journal of Cheminformatics 2017, 9, 63.
[^8]: Mitchell, J. B. O.; McDonagh, J.; Boobier, S. DLS-100 Solubility Dataset. 2017; https://risweb.st-andrews.ac.uk:443/portal/en/datasets/dls100-solubility-dataset(3a3a5abc-8458-4924-8e6c-b804347605e8).html.
[^9]: Sorkun, M. C.; Khetan, A.; Er, S. AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds. Scientific Data 2019, 6.
[^10]: Meng, J.; Chen, P.; Wahib, M.; Yang, M.; Zheng, L.; Wei, Y.; Feng, S.; Liu, W. Boosting the predictive performance with aqueous solubility dataset curation. Scientific Data 2022, 9.
[^11]:  Huuskonen, J. Estimation of Aqueous Solubility for a Diverse Set of Organic Compounds Based on Molecular Topology. Journal of Chemical Information and Computer Sciences 2000, 40, 773–777.
[^12]:  Tetko, I. V.; Tanchuk, V. Y.; Kasheva, T. N.; Villa, A. E. P. Estimation of Aqueous Solubility of Chemical Compounds Using E-State Indices. Journal of Chemical Information and Computer Sciences 2001, 41, 1488–1493.
[^13]: Bergström, C. A. S.; Wassvik, C. M.; Norinder, U.; Luthman, K.; Artursson, P. Global and Local Computational Models for Aqueous Solubility Prediction of Drug-Like Molecules. Journal of Chemical Information and Computer Sciences 2004, 44, 1477–1488.
[^14]: Rytting, E.; Lentz, K. A.; Chen, X.-Q.; Qian, F.; Venkatesh, S. Aqueous and cosolvent solubility data for drug-like organic compounds. The AAPS Journal 2005, 7, E78–E105.
[^15]: Delaney, J. S. ESOL: Estimating Aqueous Solubility Directly from Molecular Structure. Journal of Chemical Information and Computer Sciences 2004, 44, 1000–1005.
[^16]: Wassvik, C. M.; Holmén, A. G.; Bergström, C. A.; Zamora, I.; Artursson, P. Contribution of solid-state properties to the aqueous solubility of drugs. European Journal of Pharmaceutical Sciences 2006, 29, 294–305.
[^17]: Popović, G.; Čakar, M.; Agbaba, D. Acid–base equilibria and solubility of loratadine and desloratadine in water and micellar media. Journal of Pharmaceutical and Biomedical Analysis 2009, 49, 42–47.
[^18]: Forbes, G. S.; Coolidge, A. S. Relations between distribution ratio, temperature and concentration in system: water, ether, succinic acid. Journal of the American Chemical Society 1919, 41, 150–167.
[^19]: Bergström, C. A. S.; Wassvik, C. M.; Johansson, K.; Hubatsch, I. Poorly Soluble Marketed Drugs Display Solvation Limited Solubility. Journal of Medicinal Chemistry 2007, 50, 5858–5862.
[^20]: Narasimham, L.; Barhate, V. D. Kinetic and intrinsic solubility determination of some β-blockers and antidiabetics by potentiometry. Journal of Pharmacy Research 2011, 4, 532–536.

## Dependencies

### Python Libraries

It is recommended to use Python version 3.11 or higher.

```
deepchem                  2.8.0
dgl                       2.0.0+cu121
dgllife                   0.3.2
matplotlib                3.8.2
matplotlib-inline         0.1.6
notebook                  7.0.7
numpy                     1.26.3
ogb                       1.3.4
olorenchemengine          1.0.14
pandas                    2.2.0
py4j                      0.10.9.7
pytorch-lightning         1.4.1
rdkit                     2023.3.3
scikit-learn              1.4.0
scipy                     1.12.0
seaborn                   0.13.2
shap                      0.44.1
statsmodels               0.14.1
torch                     2.1.2
tqdm                      4.66.1
umap-learn                0.5.5
xgboost                   2.0.3
```

### Java Libraries

JAR files are provided in the `/cdk/lib` directory.
It is recommended to use JDK version 21 or higher.

```
cdk                       2.8.0
commons-beanutils         1.9.4
commons-collection        4.1
opencsv                   5.7.1
slf4j                     2.0.6
```

## Usage

Example workflows are provided in the `/notebook` directory.
See [here](https://docs.jupyter.org/en/latest/) if you need help with using Jupyter notebook. 

For CDK (Java) featurization, execute the following command:

```
cd cdk
<jdk\bin\java.exe> -classpath <classpath>;lib\cdk-2.8.jar;lib\commons-collections4-4.1.jar;lib\commons-beanutils-1.9.4.jar;lib\opencsv-5.7.1.jar;lib\slf4j-api-2.0.6.jar Main <dataset.csv>
```

Modify the path as needed. Note that the Java script expects the input file `<dataset.csv>` to include a column named `SMILES`.

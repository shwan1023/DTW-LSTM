# Denoising Time Window Optimized  LSTM(DTW-LSTM) Algorithm

This algorithm is designed to achieve better time series prediction effects through methods such as clustering models and denoising models, including but not limited to prediction accuracy and training costs, etc. The method has been compared with some mainstream algorithms in this field, SOTA algorithms, etc., and has proven the effectiveness of the algorithm. It also provides parameter recommendations and ablation experiments for this algorithm.

The code consists of the following parts:

1. `startup.py`: This part is what readers and practitioners should be concerned about, used for hyperparameter adjustments (such as clustering thresholds, time window lengths, data scales, etc.). It is important to note that you need to provide two sets of data to run this program, one is the original time series used for prediction, and the other is the fitting series related to this sequence (where the regression method may require you to determine according to the shape of the data you are running). Considering that in the process of conducting this experiment, the author only sought the most similar function through MATLAB's toolbox and generated the corresponding fitting series according to the function expression, this piece of code is strongly dependent on manual design in the process of dealing with specific problems, therefore it is not made public.
   - `Length`: Hyperparameter, your time window length
   - `TrainLimit`: Hyperparameter, the total length of your data
   - `threshold_pa`: Hyperparameter, your clustering threshold
   - `Fourier`: Fitting data. You can replace it with your own fitting data
   - `PekingTemp`: Original data. You can replace it with your own fitting data
2. `dataStack.py`: A special method for merging fitting series and original series to reduce abnormal noise in the original series. You can design your own fusion model. Finally, this script will return a fused data result.
3. `main.py`: The program used to start
   - `bestSeq`: The processed training set data returned, which you can use to run the LSTM network or other models and algorithms locally.
   - `bestSeq_test`: The returned test set data, unprocessed
   - `nums_Cluster`: Statistics of the number of clustering points
   - `nums_LegalTimes`: Statistics of legal time window states (i.e., how many time windows have normal features inside.

# Precautions

- The private dataset for this experiment (which has been preprocessed) is made public in the `data` folder, where `data1` is the fitting series and `data2` is the original time series. The initialization of I/O processing has been completed.
- This code does not contain a large number of Python scripts. In fact, the original experiment went through several changes. Initially, it was entirely dependent on MATLAB toolbox for experiments. In the second phase, most of the past algorithm designs were overthrown, and a Python project was reconstructed for experiments, but the experiments still relied on MATLAB toolbox for data preprocessing and LSTM training and prediction. The construction of this project belongs to the third phase, aiming to make the experimental code public while restructuring the code to ensure its readability. However, considering that this experiment does not belong to the category of innovation in deep learning architecture, only the code related to data processing is provided. You still need to build the relevant neural network locally (the basic LSTM is used in this experiment) to train and predict your model and results.
- The final output data `bestSeq` may have a few units of error compared to the expected length, but it should not exceed the error of one time window. You can adjust the error to a reasonable range, or post-process the data to reasonable figures.
- Simplifying the classification of high-dimensional spaces in clustering with Euclidean distance is a very rudimentary approach. In fact, this study can innovate in the patterns, dimensions, and norms of clustering. Considering the strong dependency between clustering algorithms and datasets, when using this code for your own research experiments to optimize clustering effects, there are several approaches:
  - First, measure the average clustering situation corresponding to the dataset object to be studied, and then design a more reasonable `threshold_pa` related to this dataset.
  - Or modify the `statusSpaceCreate` function in this code, which controls the actual clustering pattern.
  - Modify the norm `norm()` in `SingleClusterAlgorithm` and replace it with a more appropriate clustering threshold calculation method.


# File Introduction

- `data`: Contains data processed by MATLAB, saving the original and fitting sequences in the form of text files. Among them:
  - `dataFourier` and `dataPekingTemp` are our preprocessed private datasets, while the others are public datasets from Kaggle, with links provided in the paper.
  - `ComparedSOTA_Dataset` consists of predicted data for various datasets processed by the DTW-LSTM algorithm. The predictive performance will be validated against four SOTA methods using both the Mann-Whitney U Test and the Permutation Test.
  - It's important to note that these sequences are all univariate time series.
- `result`: Records the comparison results with SOTA
  - Datasets named with the prefix `processed` are training set sequences processed by DTW.
  - Datasets named with the suffix `DatasetAndResult` are the overall results obtained after training with DTW-LSTM.
  - `SOTA_Record` contains specific comparison data with SOTA, comparing our method with four SOTA methods across three datasets. The experiment proves the effectiveness of our method, with only one method showing a significant advantage over ours in one dataset.
  - Files with `.sfit` extension are programs for fitting the original series using MATLAB, i.e., curve fitters.
- `SOTA`: Records comparisons with SOTA methods. This study uses four methods as SOTA benchmarks to measure performance: AP, FF, MSTL, and TBATS.
- Files in the main program prefixed with `Pre_PostProcessing` are SOTA test files.

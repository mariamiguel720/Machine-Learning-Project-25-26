# **Cars 4 You: Expediting Car Evaluations with ML**

**Nova IMS** | Fall 2025/2026    
**Course:** Machine Learning

## **Group 40**  
- **Ana Macedo** | 20250405  
- **Catarina Mendinhas** | 20250422  
- **Lourenço Silva** | 20250453  
- **Maria Fonseca** | 20250380

### Objective
Build a robust car-price prediction model that can replicate and accelerate Cars 4 You’s current mechanic-based valuation process. The model should estimate a fair purchase price from the details a seller provides online, reducing the need for manual inspections, shortening waiting lists, and enabling faster, scalable decision-making while maintaining consistent valuation quality.

### Data Used
Cars 4 You database from 2020.

## Folders Overview
There are four folders, each containing the project guidelines, the data files and first and final deliveries of the project.

The **Final Delivery** folder is divided into 3 subfolders: 
- *Source* containing the .py files with the functions used throughout the notebooks,
- *Notebooks* with 3 jupyter notebooks,
- *Results* containing the models' results for submission in the Kaggle competition.

Focusing on the **Final Delivery Notebooks**, it includes:
- **01_EDA**: In-depth exploratory data analysis was performed, examined statistical summaries and visual representations for each feature, both individually and between features.

- **02_Preprocessing_Feature Selection_Modelling**: It encompasses comprehensive data cleaning and preprocessing, feature selection through a voting-based approach, and the development and evaluation of multiple modeling algorithms, culminating in a comparative analysis to identify the most effective solution. It was assessed how varying the strictness of feature selection using feature subsets defined by the level of agreement across multiple selection methods impacts the performance of previously identified top-performing models.

- **03_Open_Ended_Data_Preprocessing**: This section compares the performance of several tuned regression models under different preprocessing strategies and brand-based data segmentations, using a fixed feature set for consistency. Model performance is evaluated with R² and MAE to assess the impact of preprocessing choices and global versus brand-specific modeling on predictive accuracy and generalization.

Focusing on the **Final Delivery Source**, it includes:

- **data_correction_preprocessing.py**: Contains functions for correcting numerical and categorical features, handling outliers, encoding and scaling variables, imputing missing values, and a wrapper function that executes the full preprocessing pipeline.

- **feature_engineering**: Provides functions for creating and transforming features used in model development.

- **feature_selection**: Implements configurable feature selection methods, including variance filtering, correlation-based filtering, Spearman correlation, RFE, Lasso, and a voting-based comparison across methods.

- **model_and_assessment**: Includes functions for hyperparameter tuning via Randomized Search, model evaluation and comparison, exporting results for Kaggle submissions, and comparing performance across different feature sets.

- **visualizations**: Contains reusable functions for common exploratory and diagnostic visualizations, such as boxplots, heatmaps, and summaries of outliers and missing values.
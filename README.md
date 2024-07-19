# Banking Customer Churn Prediction - Understanding Customer Behavior and Predicting Churn in Banking Institutions

**Documentation is also available in Serbian in the file Dokumentacija.pdf.**

## Introduction

Welcome to my first machine learning project! This project was undertaken for academic purposes and aims to explore machine learning concepts and methodologies through practical application. The project is structured similarly to a research study, providing insights and findings based on the experiments conducted. I hope you find this work both informative and enjoyable.

## Description of the Dataset
This dataset contains information about bank customers and their churn status, which indicates whether they have exited the bank or not. It is suitable for exploring and analyzing factors influencing customer churn in banking institutions and for building predictive models to identify customers at risk of churning.

### Features:
* **RowNumber**: The sequential number assigned to each row in the dataset.
* **CustomerId**: A unique identifier for each customer.
* **Surname**: The surname of the customer.
* **CreditScore**: The credit score of the customer.
* **Geography**: The geographical location of the customer (e.g., country or region).
* **Gender**: The gender of the customer.
* **Age**: The age of the customer.
* **Tenure**: The number of years the customer has been with the bank.
* **Balance**: The account balance of the customer.
* **NumOfProducts**: The number of bank products the customer has.
* **HasCrCard**: Indicates whether the customer has a credit card (binary: yes/no).
* **IsActiveMember**: Indicates whether the customer is an active member (binary: yes/no).
* **EstimatedSalary**: The estimated salary of the customer.
* **Exited**: Indicates whether the customer has exited the bank (binary: yes/no).

## Data Preparation
Since the data has already been collected (all data is in the *Churn_Modeling.csv*), it is only necessary to prepare it for further training of potential models. All the provided data is correct—there are no missing values. As for anomalies, they are present in the clients' ages, credit scores, and the number of services each client uses. Anomalies are values that deviate from the data average. I removed them using the *IQR* *method*. The image below shows an example of the anomaly distribution by client age. Next image shows anomalies by client age (anomalies are marked with black dots):

<div align="center">
  <img loading="lazy" width="594" alt="Screenshot 2024-07-19 at 11 27 06" src="https://github.com/user-attachments/assets/bac91c2d-48bd-453e-b8db-1d28aa905026">
</div>

There are 15 anomalies in credit scores, 359 in ages, and 60 in the number of products each client has in the bank. All anomalies have been removed.

It is quite a reasonable assumption that the columns *RowNumber*, *CustomerId*, and *Surname* do not influence whether a client will leave the bank. I also doubt the columns *Geography* and *Gender*, but I wouldn't remove them at the very beginning. In the section “Selection of the Most Important Attributes,” there will be more discussion about which attributes/columns are unnecessary, i.e., which barely influence the model's output.

Since there are string-type values, they must be converted to numerical values. Through research, I concluded that the best method for encoding data is *One Hot Encoding* because it gives equal importance to all data, which is not the case with *Label Encoding*. However, I used *Label Encoding* for the *Gender* column because it has only two values, and it makes no sense to use *One Hot Encoding* for it since it will eventually be reduced to *Label Encoding*.

After encoding the data, the next step is to split them into X and y sets to start balancing and normalization. Balancing is necessary, as I confirmed by printing the number of outputs when Exited is zero versus one, finding that the ratio is 7677:1891. For balancing, I did not use *RandomOver(Under)Sampler* because it proved to be a poor balancing method. Instead, I used the *SMOTETomek* method, which combines under-sampling and over-sampling operations. I chose this balancing method because I believe it is essential that the minority data be multiplied in the sense that more similar data will be generated for those that are missing, while the majority data, i.e., more similar or identical data, will be reduced by under-sampling.

Next, the data is split into training and test sets, followed by normalization. I found many methods that can perform normalization, but I decided on *MinMaxScaler* because it scales to the range [0,1] and is also recommended when the data range is important.

## Exploratory Data Analysis
I begin the exploratory data analysis with correlation. I also performed the correlation analysis before balancing, but since it falls under exploratory data analysis, I am writing about it here. The correlation matrix looks like this: 
<div align="middle">
  <img loading="lazy" width="903" alt="Screenshot 2024-07-19 at 11 58 19" src="https://github.com/user-attachments/assets/4f6dd76a-12cd-4fda-b289-a0e8a8514d7c">
</div>

The highest correlation (excluding the diagonal) is -0.58, which is considered strong in some contexts, but since we defined a strong correlation as 0.8 and above, I won't remove anything.
Further data analysis revealed that the average age is around 39 years, the average credit score is approximately 650, the average account balance is around 76,485, and the average salary is about 100,090. The following graphs provide more information about the dataset:
<div align="middle">
  <img loading="lazy" width="735" alt="Screenshot 2024-07-19 at 12 10 45" src="https://github.com/user-attachments/assets/bc885695-ddfa-48fd-bc81-d39c7ab1d8d9">
</div>

## Models

In addition to the models we used in the faculty exercises - LinearRegression, KNeighborsClassifier, and DecisionTreeClassifier, I have added several more models:
* **GaussianNB**
* **Bagging** (Bootstrap Aggregating) - RandomForestClassifier, ExtraTreesClassifier
* **Boosting** - GradientBoostingClassifier, AdaBoostClassifier
* **Stacking** - StackingClassifier

### Bagging (Bootstrap Aggregating)
Bagging is a technique that uses multiple instances of the same model trained on different subsets of data. The goal is to reduce model variance. Characteristics: 

* Sampling with replacement - Each model is trained on different subsets of training data generated with bootstrap samples. 
* Parallel training - All models are trained independently and in parallel. 
* Combining predictions - Predictions are combined through averaging (for regression) or majority voting (for classification).

### Boosting
Boosting is a sequential technique that uses multiple weak learners, usually simple models like *Decision Trees*, where each subsequent model tries to correct the errors of the previous models. Characteristics: 
* Sequential training - Models are trained one after another, each trying to correct the errors of the previous ones. 
* Sample weighting - Samples misclassified by previous models are given higher weights.
* Combining predictions - Combination is done by weighting the predictions of all models.

### Stacking
Stacking is a technique that combines predictions from different models (base learners) using a meta-model (stacker) that makes the final decision based on these predictions. Characteristics: 
* Combination of different models - Different types of models are used (e.g., *Logistic Regression, SVM, Decision Tree*). 
* Meta-model - Predictions of the base models are used as inputs for the meta-model, which learns how to best combine these predictions. 
* Two-phase training - First, the base models are trained, then the meta-model is trained on their predictions.


### KNeighborsClassifier and Finding k
It is well known that when discussing KNeighborsClassifier, the biggest challenge is determining k. Using the Elbow method, in most cases, k is found to be three.
<div align="middle">
  <img width="689" alt="Screenshot 2024-07-19 at 12 28 14" src="https://github.com/user-attachments/assets/160cf0e0-d55f-466b-b546-5aa0c3e5288f">
</div>

## Tuning Hyperparameters of Used Models
Given that the dataset is not very small and multiple models need to be trained and the code needs to be run multiple times, and considering all the research and other work, the number of hyperparameters I tuned is not particularly large because increasing the number of parameters would also increase the execution time of the program.
I tuned the hyperparameters using *GridSearchCV*. I adjusted the following hyperparameters: 
* LogisticRegressionCV - Cs and cv 
* KNeighborsClassifier - algorithm 
* DecisionTreeClassifier - criterion
* RandomForestClassifier - n_estimators and max_depth
* ExtraTreesClassifier - n_estimators and max_features 
* StackingClassifier - cv 
* GradientBoostingClassifier - n_estimators and max_depth 
* AdaBoostClassifier - n_estimators and learning_rate

In the section “Analysis of Model Prediction Results,” I wrote about the results after tuning the hyperparameters.

## Cross-Validation
I performed cross-validation for each model using the *cross_val_score* function from the sklearn library.

## Analysis of Model Prediction Results

I conducted the analysis of the results using the following metrics: 
* **Accuracy score**
* **Precision score**
* **Recall score** 
* **F1 score** 
* **Confusion matrix** 
* **Jaccard score** - a measure of similarity between two sets. The mathematical formula is as follows: 
	* Jaccard score = (number of elements common to both sets) / (number of elements in the union of both sets)
     <div align="middle">
	    <img width="162" alt="Screenshot 2024-07-19 at 13 06 44" src="https://github.com/user-attachments/assets/51c87a75-6073-47aa-a622-d0e6d0e6bd85">
     </div>
* **Log loss score** - indicates how close the predicted probability is to the corresponding actual/true value (0 or 1 in the case of binary classification). The more the predicted probability differs from the actual value, the higher the log loss value.

Model accuracy varies from eighty to eighty-eight percent, as do precision, recall, and F1 score. The confusion matrix shows that the model misses about 500 samples in total. The Jaccard score is around 0.7, which is satisfactory, but the Log loss score is not quite right. The best value is around three, but this is not commendable since it is best for the Log loss score to be close to zero.

Of all the models, *GaussianNB* performed the worst both before and after hyperparameter tuning, while *RandomForestClassifier* performed the best considering all metric values, both with and without hyperparameter tuning.

Tuning hyperparameters yields small improvements of about one to two percent across all metrics.

The results obtained from training different models are satisfactory, and the model is usable considering that we are not predicting whether someone has a disease or not, so, for example, FN in the confusion matrix is acceptable in this case, with a value of around two hundred out of ten thousand data points. Also, looking at the other metrics, the results are fine except for the Log loss score, which could be better.

## ROC Curves
<div align="middle"><img width="309" alt="Screenshot 2024-07-19 at 13 12 56" src="https://github.com/user-attachments/assets/a575421c-1236-41be-80fd-b8fd101f2c45"></div>
The ROC curve (Receiver Operating Characteristic curve) is a graphical representation of the performance of a binary classification model, showing the relationship between the true positive rate (TPR) and the false positive rate (FPR) at various decision thresholds. The ROC curve is used to evaluate and compare the performance of classification models. TPR and FPR are calculated using the following formulas: 
<div align="middle">
  <img width="438" alt="Screenshot 2024-07-19 at 13 12 13" src="https://github.com/user-attachments/assets/7ffadff5-560a-4731-b7c1-677068bdaa80">
</div>
The ROC curves for all models are quite good and are very close to the upper left corner. On the next image you can see ROC Curve for RandomForestClassifier:
<div align="middle">
  <img width="517" alt="Screenshot 2024-07-19 at 13 17 50" src="https://github.com/user-attachments/assets/717ba2b6-2670-4ef1-89ce-37321a8ba9f4">
</div>

## Selection of the Most Important Attributes
To select the most important attributes, I used two different algorithms: *SelectKBest* and *RFE*. *SelectKBest* consistently chooses '*Gender*', '*Age*', '*Tenure*', '*Balance*', '*NumOfProducts*', '*HasCrCard*', '*IsActiveMember*', '*Geography_France*', '*Geography_Germany*', and '*Geography_Spain*', while *RFE* selects '*CreditScore*', '*Gender*', '*Age*', '*Tenure*', '*Balance*', '*NumOfProducts*', '*IsActiveMember*', '*EstimatedSalary*', '*Geography_France*', and '*Geography_Spain*'.

These two algorithms differ by only two or three attributes. When running the models, the results are approximately the same as when the attributes are not reduced.

I mentioned at the beginning that I doubted the '*Gender*' and '*Geography*' columns, but now I no longer doubt them and consider these columns to be very important since the algorithms for selecting the most important attributes did not eliminate them.

I considered training the model on all attributes in the end, but that could lead to training problems and worse results, as the model might consider the '*RowNumber*' column to be important, which it certainly is not because the value is unique for each person.

## Conclusion
Considering their purpose, the models provide satisfactory results and I consider them usable, especially when the model with the best metric results is selected. The attributes that were removed at the very beginning, '*RowNumber*', '*Surname*', and '*CustomerId*', definitely should remain removed, and further reduction of attributes is unnecessary because it gives slightly worse results, and collecting these attributes is simple since if all other attributes about a client are being collected, these will be too, as they are important for the bank to know.


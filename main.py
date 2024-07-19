import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, jaccard_score, log_loss, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import warnings
from sklearn.feature_selection import SelectKBest, f_classif, RFE



warnings.simplefilter(action='ignore')

#Loading data------------------------------
data = pd.read_csv('Churn_Modelling.csv')
#------------------------------------------


# Checking missing values-----------------
print("Checking missing values: ")
print(data.isnull().sum())
print('----------------------------------')
#------------------------------------------


# Histograms, pies, graphs...--------------
plt.figure(figsize=(8, 6))
sns.histplot(data['Age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.figure(figsize=(8, 6))
sns.histplot(data['CreditScore'], bins=20, kde=True)
plt.title('Distribution of CreditScore')
plt.xlabel('CreditScore')
plt.ylabel('Frequency')

plt.figure(figsize=(8, 6))
data['Geography'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Geography Distribution')
plt.ylabel('')

plt.figure(figsize=(8, 6))
data['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Gender Distribution')
plt.ylabel('')

plt.figure(figsize=(8, 6))
data['IsActiveMember'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Membership Distribution')
plt.ylabel('')

plt.figure(figsize=(8, 6))
data['HasCrCard'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('HasCrCard Distribution')
plt.ylabel('')

plt.figure(figsize=(10, 6))
sns.countplot(x='Gender', hue='Exited', data=data)
plt.title('Exited Cases by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')

plt.figure(figsize=(10, 6))
sns.countplot(x='HasCrCard', hue='Exited', data=data)
plt.title('Exited Cases by HasCrCard')
plt.xlabel('HasCrCard')
plt.ylabel('Count')

print("Average age: ", data['Age'].mean())
print("Average credit score: ", data['CreditScore'].mean())
print("Average balance: ", data['Balance'].mean())
print("Estimated salary: ", data['EstimatedSalary'].mean())

#------------------------------------------


#Removing anomalies------------------------

numeric_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

def count_outliers_and_get_bounds(column_data):
    Q1 = column_data.quantile(0.25)
    Q3 = column_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)]
    return len(outliers), lower_bound, upper_bound

outliers_count = {}
outlier_bounds = {}

for column in numeric_columns:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=data[column])
    plt.title(f'Boxplot of {column}')
    plt.show()
    
    outliers_count[column], lower_bound, upper_bound = count_outliers_and_get_bounds(data[column])
    outlier_bounds[column] = (lower_bound, upper_bound)

for column, count in outliers_count.items():
    print(f'{column}: {count} outliers')

for column, (lower_bound, upper_bound) in outlier_bounds.items():
    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

print(f'Number of rows after deleting anomalies: {len(data)}')
#------------------------------------------


# Remoivng unnecessary columns-------------
data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
#------------------------------------------


# One-Hot Encoding for 'Geography' and LabelEncoder for 'Gender'
data = pd.get_dummies(data, columns=['Geography'])
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data = data.astype(int)
#------------------------------------------


# Correlation matrix-----------------------
corr_matrix = data.corr()
plt.figure(figsize=(14,10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
#------------------------------------------


# Is dataset unbalanced?
counter = Counter(data['Exited'])
print(counter)
#------------------------------------------


# Dividing data to X and y--------
y = data['Exited']
X = data.drop('Exited', axis=1)
smote_tomek = SMOTETomek(random_state=42)
X, y= smote_tomek.fit_resample(X,y)

print('Resampled dataset shape:', Counter(y))
#------------------------------------------

# Dividing data to train and test and normalization
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, stratify=y)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#------------------------------------------

# Creating and training models with hyperparams tuning and cross validation

estimators=[
    ('svc', SVC(probability=True, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators = 100, random_state = 42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
]

models = [
    LogisticRegressionCV(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),

    GaussianNB(),

    RandomForestClassifier(),
    ExtraTreesClassifier(),

    StackingClassifier(estimators=estimators,final_estimator=LogisticRegressionCV()),

    GradientBoostingClassifier(),
    AdaBoostClassifier()
]

hiperparams = {
    'LogisticRegressionCV': {
        'Cs': [1, 10, 100],
        'cv': [1, 2, 5, 10]
    },
    'KNeighborsClassifier': {
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    'DecisionTreeClassifier': {
        'criterion': ['gini', 'entropy', 'log_loss']
    },
    'RandomForestClassifier': {
        'n_estimators': [10,50, 100, 200],
        'max_depth': [None, 10, 25, 50],
    },
    'ExtraTreesClassifier':{
        'n_estimators': [100,200],
        'max_features': [0.1, 0.2, 'sqrt', 'log2']
    },
    'StackingClassifier': {
        'cv': [3, 5]
    },
    'GradientBoostingClassifier': {
        'n_estimators': [75, 100], 
        'max_depth': [3, 5] 
    },
    'AdaBoostClassifier': {
        'n_estimators': [50, 100],
        'learning_rate': [0.1, 1.0]
    }
}

def model_performance(name, model, X_train, X_test, y_train, y_test, y_pred):

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    #New metrics
    jaccard = jaccard_score(y_test,y_pred)
    log = log_loss(y_test,y_pred)

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("Confusion matrix: \n", confusion)

    #New metrics
    print("Jaccard score:", jaccard)
    print("Log loss: ", log)

    #ROC curve
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    auc = roc_auc_score(y_test, y_prob)

    #Cross validation
    cv_score = cross_val_score(model, X_train, y_train, cv = 5)
    print("Cross validation: ", cv_score)
    print("Average precision of cross validation: ", cv_score.mean())

    if name == 'KNeighborsClassifier' or name == 'KNeighborsClassifier with hyperparams':
            error_rate = []

            k_values = range(1,13,2)
            for k in k_values:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                error_rate.append(1 - accuracy_score(y_test, y_pred))
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))

            ax1.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
            ax1.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title('ROC Curve')
            ax1.legend(loc="lower right")

            ax2.plot(k_values, error_rate, color='blue', linestyle='dashed', marker='o',
                    markerfacecolor='red', markersize=10)
            ax2.set_title('Error Rate vs. K Value')
            ax2.set_xlabel('K')
            ax2.set_ylabel('Error Rate')

            ax3.text(0.5,0.8,name, horizontalalignment='center', verticalalignment='center')
            ax3.text(0, 0.7, 'Accuracy: ' + str(accuracy),  horizontalalignment='left', verticalalignment='top')
            ax3.text(0, 0.6, 'Precision: ' + str(precision),  horizontalalignment='left', verticalalignment='top')
            ax3.text(0, 0.5, 'Recall: ' + str(recall),  horizontalalignment='left', verticalalignment='top')
            ax3.text(0, 0.4, 'F1:' + str(f1), horizontalalignment='left', verticalalignment='top')
            ax3.text(0, 0.3, 'Confusion matrix: ',  horizontalalignment='left', verticalalignment='top')
            ax3.text(0, 0.2, str(confusion),  horizontalalignment='left', verticalalignment='top')
            ax3.text(0, 0.1, 'Jaccard score: ' + str(jaccard),  horizontalalignment='left', verticalalignment='top')
            ax3.text(0, 0, 'Log loss: ' + str(log),  horizontalalignment='left', verticalalignment='top')

            ax3.axis('off') 

            plt.tight_layout()
            plt.show()
            best_k = k_values[np.argmin(error_rate)]
            print(f'The best k value is: {best_k}')

    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        ax1.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend(loc="lower right")

        ax2.text(0.5,0.8,name, horizontalalignment='center', verticalalignment='center')
        ax2.text(0, 0.7, 'Accuracy: ' + str(accuracy),  horizontalalignment='left', verticalalignment='top')
        ax2.text(0, 0.6, 'Precision: ' + str(precision),  horizontalalignment='left', verticalalignment='top')
        ax2.text(0, 0.5, 'Recall: ' + str(recall),  horizontalalignment='left', verticalalignment='top')
        ax2.text(0, 0.4, 'F1:' + str(f1), horizontalalignment='left', verticalalignment='top')
        ax2.text(0, 0.3, 'Confusion matrix: ',  horizontalalignment='left', verticalalignment='top')
        ax2.text(0, 0.2, str(confusion),  horizontalalignment='left', verticalalignment='top')
        ax2.text(0, 0.1, 'Jaccard score: ' + str(jaccard),  horizontalalignment='left', verticalalignment='top')
        ax2.text(0, 0, 'Log loss: ' + str(log),  horizontalalignment='left', verticalalignment='top')

        ax2.axis('off') 

        plt.tight_layout()
        plt.show()

def go_trough_all_models(X_train, X_test, y_train, y_test, hyperOn):
    for model in models:
        name = model.__class__.__name__

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("--------------------",name,"--------------------")
        model_performance(name, model, X_train, X_test, y_train,y_test,y_pred)
        #Hyperparams tuning
        if hyperOn:
            model_h = hiperparams.get(name, {})
            if model_h:
                grid = GridSearchCV(model, model_h, cv = 5)
                grid.fit(X_test, y_test)
                print("--------------------------------------------------")
                print("The best parameters: ", grid.best_params_)

                model = grid.best_estimator_
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                model_performance(name + ' with hyperparams', model, X_train, X_test, y_train, y_test, y_pred)
            else:
                print('Model has not any params for optimization')

        print('\n\n')

go_trough_all_models(X_train, X_test, y_train, y_test, True)
#------------------------------------------


#Selection of the Most Important Attribute ---------------

print(data.drop('Exited',axis=1).columns)
original_feature_names = data.drop('Exited',axis=1).columns

#SelectKBest
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_indices = selector.get_support(indices=True)

selected_feature_names = [original_feature_names[i] for i in selected_indices]

print("Selected columns with SelectKBest:")
print(selected_feature_names)

go_trough_all_models(X_train_selected, X_test_selected, y_train, y_test, False)

#RFE
estimator = RandomForestClassifier()
rfe = RFE(estimator, n_features_to_select=10, step=1)

X_train_selected = rfe.fit_transform(X_train, y_train)
X_test_selected = rfe.transform(X_test)

selected_indices = np.where(rfe.support_)[0]

selected_feature_names = [original_feature_names[i] for i in selected_indices]

print("Selected columns with RFE:")
print(selected_feature_names)

go_trough_all_models(X_train_selected, X_test_selected, y_train, y_test, False)



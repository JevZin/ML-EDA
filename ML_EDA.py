import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder


# This gets the number of unique values in every categorical feature column

def get_unique_values(dataframe, col_list):
    for col in col_list:
        print('Column:', col)
        print('Unique values:', dataframe[col].unique())
        print('Number of each unique value:', dataframe[col].value_counts())
        print('\n')


# This gives a statistical description of numerical features according to column
def stat_description(dataframe, col_list):
    for col in col_list:
        print('Column:', col)
        print(f'Statistical description:\n{dataframe[col].describe()}')
        print('\n')


# This creates pie charts of different features according to columns
def pie_chart(dataframe, col_list):
    for i in col_list:
        plt.pie(dataframe[i].value_counts(), labels=dataframe[i].value_counts().index, autopct='%1.1f%%')
        plt.title(f'Pie chart of {i}')
        plt.show()


# This creates histograms of different features according to columns
def histogram(dataframe, col_list):
    for i in col_list:
        plt.hist(dataframe[i], edgecolor='black')
        plt.title(f'Histogram of {i}')
        plt.xlabel(i)
        plt.ylabel('Count')
        plt.show()


# This creates a bar plot that groups two features and shows each features count
def group_plot(dataframe, x, y):
    x_data = dataframe[x]
    y_date = dataframe[y]
    grouping = dataframe.groupby([x_data, y_date]).size().unstack().plot(kind='bar')
    for container in grouping.containers:
        grouping.bar_label(container)
    plt.title(f'{y} count by {x}')
    plt.ylabel(y)
    plt.tight_layout()
    plt.show()


# This creates scatter plots of different features according to columns
def scatter(dataframe, x, y):
    x_data = dataframe[x]
    y_data = dataframe[y]
    plt.scatter(x_data, y_data)
    plt.title(f'Scatterplot of {x} by {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()


# This creates scatter plots of different features hued by a third feature
def scatter_hue(dataframe, x, y, z):
    x_data = dataframe[x]
    y_data = dataframe[y]
    z_data = dataframe[z]
    sns.scatterplot(data=dataframe, x=x_data, y=y_data, hue=z_data)
    plt.tight_layout()
    plt.show()


# This calculates the train/test accuracy, precision, recall and F1 scores for the model
def metrics(y_train, y_train_pred, y_test, y_test_pred):
    confmat = confusion_matrix(y_test, y_test_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_pre = precision_score(y_train, y_train_pred)
    test_pre = precision_score(y_test, y_test_pred)
    train_rec = recall_score(y_train, y_train_pred)
    test_rec = recall_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)

    print('\nConfusion matrix: \n', confmat, '\n')
    print(f'Model train/test accuracy ' f'{train_acc:.4f}/{test_acc:.4f}')
    print(f'Model train/test precision ' f'{train_pre:.4f}/{test_pre:.4f}')
    print(f'Model train/test recall ' f'{train_rec:.4f}/{test_rec:.4f}')
    print(f'Model train/test F1 score ' f'{train_f1:.4f}/{test_f1:.4f}')
    print(f'Model train/test ROC-AUC score ' f'{train_auc:.4f}/{test_auc:.4f}')


# This calculates various metrics for regression problems
def metrics_reg(y_train, y_train_pred, y_test, y_test_pred):
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print(f'MSE train: {mse_train:.3f}')
    print(f'MSE test: {mse_test:.3f}')
    print(f'MAE train: {mae_train:.3f}')
    print(f'MAE test: {mae_test:.3f}')
    print(f'R2 train: {r2_train:.3f}')
    print(f'R2 test: {r2_test:.3f}')


# This generates a heatmap of features in a dataframe
def heatmap(dataframe):
    df_encoded = dataframe.copy()
    label_encoders = {}

    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le

    cov_matrix = df_encoded.cov()
    cov_matrix.columns = dataframe.columns
    cov_matrix.index = dataframe.columns
    plt.figure(figsize=(20, 20))
    sns.heatmap(cov_matrix, annot=True, cmap='copper', fmt='.2f', linewidths=0.5)
    plt.title('Covariance matrix heatmap')
    plt.tight_layout()
    plt.show()


# This plots the learning curves of a model
def learning_curves(estimator, x, y):
    train_sizes, train_scores, test_scores = learning_curve(estimator=estimator, X=x, y=y,
                                                            train_sizes=np.linspace(0.1, 1.0, 10),
                                                            cv=10, n_jobs=1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5,
             label='Validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.grid()
    plt.xlabel('Number of training examples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.03])
    plt.show()

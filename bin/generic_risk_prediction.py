import pandas as pd
import build_dataset
from __init__ import *
from sklearn import metrics, linear_model, svm
from sklearn.ensemble import RandomForestClassifier


def calculate_metrics(true_values, predicted_values):
    """
    Calculate precision, recall, f1-score and support based on classifier output

    :param true_values: actual class output
    :param predicted_values: predicted class output
    :return: calculated metrics
    """

    return metrics.classification_report(y_true=true_values, y_pred=predicted_values)


def logistic_model(train, test):
    """
    Logistic Regression model for classification

    :param train: training dataset
    :param test: testing dataset
    """

    log_model = linear_model.LogisticRegression()

    log_model.fit(X=train[train.columns[:-1]],
                  y=train['Tumor'])

    predicted = log_model.predict(X=test[test.columns[:-1]])
    predicted = pd.DataFrame(predicted)

    logger.info('Calculated metrics')
    logger.info(calculate_metrics(test['Tumor'], predicted) + '\n')


def random_forest_model(train, test):
    """
    Random Forest model for classification

    :param train: training dataset
    :param test: testing dataset
    """

    forest_model = RandomForestClassifier(n_estimators=100)

    forest_model.fit(X=train[train.columns[:-1]].as_matrix(),
                     y=train['Tumor'].as_matrix())

    predicted = forest_model.predict(X=test[test.columns[:-1]])

    logger.info('Calculated metrics')
    logger.info(calculate_metrics(test['Tumor'], predicted) + '\n')


def svm_model(train, test):
    """
    Support Vector Machine model for classification

    :param train: training dataset
    :param test: testing dataset
    """

    svc_model = svm.SVC()

    svc_model.fit(X=train[train.columns[:-1]].as_matrix(),
                  y=train['Tumor'].as_matrix())

    predicted = svc_model.predict(X=test[test.columns[:-1]])

    logger.info('Calculated metrics')
    logger.info(calculate_metrics(test['Tumor'], predicted) + '\n')


def main():
    """
    Execute generic classification methods on DNA methylation data

    :return: metrics of each classifier
    """
    
    train, test = build_dataset.main()

    logger.info('Applying Logistic Regression')
    logistic_model(train, test)

    logger.info('Applying Random Forest')
    random_forest_model(train, test)

    logger.info('Applying Support Vector Machine')
    random_forest_model(train, test)


if __name__ == '__main__':
    main()

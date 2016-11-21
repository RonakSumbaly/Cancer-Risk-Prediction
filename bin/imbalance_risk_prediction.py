import pandas as pd
import build_dataset
from __init__ import *
from imblearn.combine import SMOTETomek, SMOTEENN
from classifiers import logistic_model, random_forest_model, naive_bayes_model

def over_under_sampling(data):
    column_names = data.columns[:-1]
    smote_tomek = SMOTEENN(ratio='auto')
    features, label = smote_tomek.fit_sample(data[data.columns[:-1]], data['Tumor'].as_matrix())

    data = pd.DataFrame(features)
    data.columns = column_names
    data['Tumor'] = label

    logger.info(data)
    return data

def main():
    train, test = build_dataset.main()
    train, test = build_dataset.normalize_datasets(train, test)
    train = over_under_sampling(train)


    logger.info('Applying Logistic Regression')
    logistic_model(train, test)

    logger.info('Applying Random Forest')
    random_forest_model(train, test)

    logger.info('Applying Gaussian Naive Bayes')
    naive_bayes_model(train, test)

if __name__ == '__main__':
    main()
def nb_classfication(X, y):
    '''
        using naive bayes classifier to classfy the cancer type using the
        feature X as the input and y as the true label
        return the metric on each cv fold as a dict
    '''
    
    from sklearn.model_selection import cross_validate
    from sklearn.naive_bayes import GaussianNB
    import pandas as pd
    
    cv_performance_dict = dict()

    # Initialize Naive Bayes classifier
    naive_bayes = GaussianNB()

    scoring = {'accuracy': 'accuracy'}

    # Perform stratified 5-fold cross-validation with multiple metrics
    cv_results = cross_validate(naive_bayes, X, y, cv=5, scoring=scoring)

    # Print cross-validation results
    print("Cross-validation results:")
    for metric in scoring:
        cv_performance_dict[metric] = cv_results['test_' + metric]
    
    df = pd.DataFrame(cv_performance_dict)
    df.index = ['fold' + str(i+1) for i in df.index]
    
#     mean_value = pd.DataFrame(df.mean()).T
#     mean_value.index = ['avg']
#     std_value = pd.DataFrame(df.std()).T
#     std_value.index = ['std']

#     cv_performance_df = pd.concat([df, mean_value, std_value], axis=0)
    
    
    return df
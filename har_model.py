import constants 
import itertools
from re import L
import pickle 
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics 
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

sns.set_style('darkgrid')
plt.rcParams['font.family'] = 'Times New Roman'

labels = constants.label_dict.values()

def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df

train_data = read_csv(constants.train_csv)
test_data = read_csv(constants.test_csv) 

def divide_data(data):
    x = data.drop(['subject', 'Activity', 'ActivityName'], axis=1)
    y = data['ActivityName']
    return x, y  

train_x, train_y = divide_data(train_data)
test_x, test_y = divide_data(test_data)

def cm_graph(cm, classes, normalize=False, heading_name='CONFUSION MATRIX', cmap=plt.cm.Reds_r):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(heading_name) 
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def model_performance(model, X_train, y_train, X_test, y_test, class_labels, results_path, print_cm=True, cm_cmap=plt.cm.Greens):
    
    results = dict()
    # Model Training Starts here
    training_initiated = datetime.now()
    model.fit(X_train, y_train)
    training_ended = datetime.now()
    results['training_time'] =  training_ended - training_initiated
    print('training_time {}\n\n'.format(results['training_time']))
    # Prediction on Test Data
    pred_started = datetime.now()
    y_pred = model.predict(X_test)
    pred_end = datetime.now() 
    results['pred_time'] = pred_end - pred_started
    print('pred time - {}'.format(results['pred_time']))
    results['predicted_output'] = y_pred

    r_df = pd.DataFrame(results) 
    r_df.to_csv(results_path, index = False)
   
    # Overall Model Performance 
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    # store accuracy in results
    results['accuracy'] = accuracy
    print('Accuracy: {}'.format(accuracy))
    
    # confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm
    if print_cm: 
        print('Confusion Matrix \n{}'.format(cm))
        
    # plot confusin matrix
    plt.figure(figsize=(10 , 10))
    plt.grid(b=False)
    cm_graph(cm, classes=class_labels, normalize=True, heading_name='Normalized confusion matrix', cmap = cm_cmap)
    plt.show()
    
    # get classification report
    classification_report = metrics.classification_report(y_test, y_pred)
    # store report in results
    results['classification_report'] = classification_report
    print(classification_report)
    
    # add the trained  model to the results
    results['model'] = model
    
    return results
    
def print_grid_search_attributes(model):
    print('Best Model {}'.format(model.best_estimator_))
    print('Parameters of best estimator : {}'.format(model.best_params_))
    print('Total CV sets: {}'.format(model.n_splits_))
    print('Avg CV scores of best model : {}'.format(model.best_score_))

## Random Forest Model 
parameters = {'n_estimators': np.arange(10,201,20), 'max_depth':np.arange(3,15,2)}
randomforest_classifier = RandomForestClassifier()
randomforest_grid = GridSearchCV(randomforest_classifier, param_grid=parameters, n_jobs=-1)
randomforest_results = model_performance(randomforest_grid, train_x, train_y, test_x, test_y, class_labels=labels,  results_path = constants.log_results_path)

plt.figure(figsize=(8,8))
plt.grid(b=False)
cm_graph(randomforest_results['confusion_matrix'], classes=labels, cmap=plt.cm.Greens, )
plt.show()

print_grid_search_attributes(randomforest_results['model'])

# Gradient Boosting Model
kfold_cv = KFold(n_splits = 10, shuffle = True)
gb_tree = GradientBoostingClassifier()
param_grid = {'max_depth': np.arange(5,8,2), 'n_estimators':np.arange(130,170,30)}

gb_tree_grid = GridSearchCV(gb_tree, param_grid=param_grid, n_jobs=-1)
gb_tree_grid_results = model_performance(gb_tree_grid, train_x, train_y, test_x, test_y, class_labels = labels, results_path = constants.boosting_results_path)
print_grid_search_attributes(gb_tree_grid_results['model'])


def gradient_model():
    gb_tree_new = GradientBoostingClassifier(max_depth=5, n_estimators=130) 
    gb_tree_new.fit(train_x, train_y)
    predictions_new = gb_tree_new.predict(test_x)

    print("Gradient Boosting Confusion Matrix:")
    print(confusion_matrix(test_y, predictions_new))

    print("Gradient Boosting Classification Report")
    print(classification_report(test_y, predictions_new)) 

    pickle.dump(gb_tree_new, open('models/gb_classifier.sav', 'wb'))

def random_forest_model():
    randomforest_new = RandomForestClassifier( max_depth=7, n_estimators=170) 
    randomforest_new.fit(train_x, train_y)
    predictions_new = randomforest_new.predict(test_x)

    print("Random Forest Confusion Matrix:")
    print(confusion_matrix(test_y, predictions_new))

    print("Random Forest Classification Report")
    print(classification_report(test_y, predictions_new)) 

    pickle.dump(randomforest_new, open('models/rf_classifier.sav', 'wb')) 

gradient_model()
random_forest_model()

# load_model = pickle.load(open('models/rf_classifier.sav', 'rb'))
# result = load_model.score(X_test, Y_test)
# load_model.predict(input_data)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import constants 

sns.set_style('darkgrid')
plt.rcParams['font.family'] = 'Times New Roman'

with open(constants.features_path) as text_file:
    inp_features = [l.split()[1] for l in text_file.readlines()]
    print('Total number of Features extracted are: {}'.format(len(inp_features)))

def data_processing(input_csv, sub_csv, features, out_csv):
    df = pd.read_csv(input_csv, header=None, delim_whitespace = True)
    df.columns = features
    df['subject'] = pd.read_csv(sub_csv, header=None, squeeze=True)

    y = pd.read_csv(out_csv, squeeze=True, names=['Activity'])
    y_labels = y.map(constants.label_dict)

    new_df = df
    new_df['Activity'] = y
    new_df['ActivityName'] = y_labels

    return new_df

train = data_processing(constants.input_train_data_path, constants.train_subj, inp_features, constants.output_train_data_path)
test =  data_processing(constants.input_test_data_path, constants.test_subj, inp_features, constants.output_test_data_path)

def analyse_data(inp, op):
    findings = {'Train':[sum(inp.duplicated()), inp.isnull().values.sum()],
                'Test':[sum(op.duplicated()), op.isnull().values.sum()] 
               }

    findings_df = pd.DataFrame(findings, index = ['Duplicates', 'Null values'])

    return findings_df

print(analyse_data(train, test))

def user_data():
    plt.figure(figsize=(7,7))
    plt.title('User Level Data', fontsize=20)
    plt.legend(loc ="best")
    sns.countplot(x='subject', hue='ActivityName', data = train)
    plt.show()

def label_distribution():
    plt.title('Label Count')
    sns.countplot(train['ActivityName'])
    plt.xticks(rotation=90)
    plt.show()

def process_features():
    feature_name = train.columns
    feature_name = feature_name.str.replace('[()]','')
    feature_name = feature_name.str.replace('[-]', '')
    feature_name = feature_name.str.replace('[,]','')
    train.columns = feature_name
    test.columns = feature_name

    train.to_csv('original/UCI HAR Dataset/final_dataset/train.csv', index=False)
    test.to_csv('original/UCI HAR Dataset/final_dataset/test.csv', index=False)

def acceleration_monitoring(colA, colB, dataset):
    plt.figure(figsize=(7,7))
    sns.boxplot(x = colA, y = colB, showfliers=False, data = dataset, saturation=1)
    plt.ylabel('Monitoring Acceleration')
    plt.axhline(y=-0.9, xmin=0.05, xmax=0.95, dashes=(4,4), c='r') 
    plt.axhline(y=-0.04, xmin=0.3, dashes=(5,5), c='b')
    plt.xticks(rotation=90)
    plt.show()

user_data()
label_distribution()
process_features()
acceleration_monitoring(colA = 'ActivityName', colB = 'tBodyAccMagmean', dataset = train)



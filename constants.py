
train_csv = 'original/UCI HAR Dataset/final_dataset/train.csv'
test_csv = 'original/UCI HAR Dataset/final_dataset/test.csv' 

features_path = 'original/UCI HAR Dataset/features.txt'
train_subj = 'original/UCI HAR Dataset/train/subject_train.txt'
test_subj = 'original/UCI HAR Dataset/test/subject_test.txt'
input_train_data_path = 'original/UCI HAR Dataset/train/x_train.txt'
output_train_data_path = 'original/UCI HAR Dataset/train/y_train.txt'
input_test_data_path = 'original/UCI HAR Dataset/test/X_test.txt'
output_test_data_path = 'original/UCI HAR Dataset/test/y_test.txt'

log_results_path = 'original/UCI HAR Dataset/results/log_results.csv'
boosting_results_path = 'original/UCI HAR Dataset/results/boosting_results.csv'

label_dict = {1:'WALKING', 2:'WALKING_UPSTAIRS', 3:'WALKING_DOWNSTAIRS', 4:'SITTING', 5:'STANDING', 6:'LAYING'}
labels = label_dict.values()
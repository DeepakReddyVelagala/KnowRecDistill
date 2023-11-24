import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv(r'F:\KnowRecDistill\dataset\ratings.csv')

# Use LabelEncoder to ensure that the user IDs and movie IDs are contiguous integers
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

data['userId'] = user_encoder.fit_transform(data['userId'])
data['movieId'] = movie_encoder.fit_transform(data['movieId'])

# Ensure that each user has at least K interactions in the training set
K = 10  # or whatever value you choose
user_counts = data['userId'].value_counts()
enough_data_users = user_counts[user_counts >= K].index
data = data[data['userId'].isin(enough_data_users)]

# Split the data into training, validation, and testing sets
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Save the processed data to CSV files
dataset_path = r'F:\KnowRecDistill\dataset\\'
train_data.to_csv(dataset_path+'train.csv', index=False)
val_data.to_csv(dataset_path+'val.csv', index=False)
test_data.to_csv(dataset_path+'test.csv', index=False)
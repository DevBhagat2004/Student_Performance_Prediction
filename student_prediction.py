import pandas as pd
from sklearn.model_selection import train_test_split # for splitting datasets for training and testing
from sklearn.linear_model import LogisticRegression # for logistic regression model
from sklearn.metrics import accuracy_score, confusion_matrix # for evaluating the model
from sklearn.preprocessing import LabelEncoder # for encoding categorical variables
import matplotlib.pyplot as plt # for plotting graphs

df = pd.read_csv('test_data.csv')# load the dataset

# calculate the average score & add it to dataframe
df['average_score'] = df[['math score','reading score','writing score']].mean(axis=1)

# if needs support or not
df['needs_support'] = (df['average_score']<60).astype(int)

# Dropping the columns that are not needed for the model
df = df.drop(columns=['average_score'])

# Model cant handle string values, so we need to encode categorical variables 
label_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# Create a dataset without the 'needs_support' column
x= df.drop(columns=['needs_support'])
# Create a data with only the 'needs_support' column
y = df['needs_support']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.5, random_state=42)

# Create a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

#check the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
#confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print the accuracy and confussion matrix
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)

# show the number of total predictions
num_of_total_predictions = len(y_pred)
print("Total number of predictions:", num_of_total_predictions)


# Plotting the confusion matrix
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(ticks=[0, 1], labels=['No Support', 'Needs Support'])
plt.yticks(ticks=[0, 1], labels=['No Support', 'Needs Support'])
plt.colorbar()
plt.show()



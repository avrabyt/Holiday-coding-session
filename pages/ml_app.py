import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Convert the data to a Pandas DataFrame
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

# Display the DataFrame in the Streamlit app
st.dataframe(df)

# Use Seaborn to plot a pair plot of the first 10 features
sns.pairplot(df[df.columns[:10].tolist() + ['target']], hue='target')

# Display the plot in the Streamlit app
st.pyplot()

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier on the training set
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make predictions on the test set
predictions = clf.predict(X_test)

# Calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)

# Display the accuracy in the Streamlit app
st.success(f'Accuracy: {accuracy:.2f}')

# Build the Streamlit app
st.sidebar.title("Breast Cancer Classifier")

# Create a slider for each column in the DataFrame
user_input = []
for col in df.columns[:-1]:
    slider_input = st.sidebar.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    user_input.append(slider_input)

# Make a prediction based on the user's input
prediction = clf.predict([user_input])

# Display the prediction
if prediction == 0:
    st.markdown('### The tumor is predicted to be benign.')
else:
    st.markdown('### The tumor is predicted to be malignant.')

# Calculate the false positive rate and true positive rate
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, predictions)

# Plot the ROC curve
import matplotlib.pyplot as plt
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# Display the plot in the Streamlit app
st.pyplot()



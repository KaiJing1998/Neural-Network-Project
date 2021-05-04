import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# import data
col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
             'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(r'C:\Users\KAI JING\Downloads\diabetes.csv', header=0, names=col_names)
print(data)

# feature selection
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[feature_cols]
X = data.drop("Outcome", axis=1)
# Target variable
y = data.Outcome

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print("Training Data :", X_train.shape)
print("Testing Data : ", X_test.shape)




#mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
#mlp.fit(X_train, y_train.values.ravel())

#predictions = mlp.predict(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

print("Training Prediction :")
print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))

print("Testing Prediction :")
print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))
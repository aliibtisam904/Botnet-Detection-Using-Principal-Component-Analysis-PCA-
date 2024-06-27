#Botnet Detection using Principal Component Analysis (PCA):
#Objective: Detect botnet activities in network traffic data using principal component analysis.
#Week 1: Introduction to PCA and dimensionality reduction techniques.
#Week 2: Apply PCA to reduce the dimensionality of network traffic data, extract principal components,
#and develop algorithms for botnet detection based on the reduced feature space.
#Reference: "Data Mining: Concepts and Techniques" by Jiawei Han, Micheline Kamber, and Jian Pei.
#Python Implementation: Implement PCA for dimensionality reduction using scikit-learn and apply it to
#botnet detection on network traffic data.
#Reference: scikit-learn documentation (https://scikit-learn.org/stable/documentation.html).
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def create_data():
    # Create a dictionary with network traffic data
    data = {'Duration': [1, 2, 3, 4, 5],
            'Source IP': ['192.168.1.1', '192.168.1.2', '192.168.1.3', '192.168.1.4', '192.168.1.5'],
            'Destination IP': ['192.168.1.6', '192.168.1.7', '192.168.1.8', '192.168.1.9', '192.168.1.10'],
            'Source Port': [80, 8080, 443, 8080, 80],
            'Destination Port': [8080, 80, 8080, 443, 80],
            'Protocol': [1, 2, 1, 2, 1],
            'Packets': [100, 200, 300, 400, 500],
            'Bytes': [1000, 2000, 3000, 4000, 5000],
            'Label': ['Normal', 'Botnet', 'Normal', 'Botnet', 'Normal']}
    df = pd.DataFrame(data)
    df.to_csv('network_traffic_data.csv', index=False)

def load_data():
    # Load network traffic data from file
    df = pd.read_csv('network_traffic_data.csv')
    if df.empty:
        raise Exception("No data found in the file.")
    return df

def preprocess_data(df):
    # Split data into features and labels
    X = df.drop('Label', axis=1)
    y = df['Label']

    # Encode string values to numerical values
    le = LabelEncoder()
    X['Source IP'] = le.fit_transform(X['Source IP'])
    X['Destination IP'] = le.fit_transform(X['Destination IP'])

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def apply_pca(X_train, X_test):
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

def train_model(X_train, y_train):
    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Print classification report
    print(classification_report(y_test, y_pred, zero_division=1))

    # Print confusion matrix
    print(confusion_matrix(y_test, y_pred))
    return y_pred

def visualize_pca(X_train_pca, y_train):
    # Convert labels to numerical values
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    # Visualize PCA-transformed data
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_encoded)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA Scatter Plot')
    plt.show()

def visualize_confusion_matrix(y_test, y_pred):
    # Visualize confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    create_data()
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    X_train_pca, X_test_pca = apply_pca(X_train, X_test)
    visualize_pca(X_train_pca, y_train)
    model = train_model(X_train_pca, y_train)
    y_pred = evaluate_model(model, X_test_pca, y_test)
    visualize_confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    main()
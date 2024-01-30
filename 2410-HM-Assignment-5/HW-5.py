import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)


def knn_classification():
    # Load Iris dataset
    iris = datasets.load_iris()
    iris_data = iris.data
    iris_labels = iris.target

    # Set a random seed for reproducibility
    np.random.seed(int(time.time()))

    # Permute the indices for randomization
    indices = np.random.permutation(len(iris_data))

    # Number of training samples
    n_training_samples = 19

    # Separate the dataset into training and test sets
    learnset_data = iris_data[indices[:-n_training_samples]]
    learnset_labels = iris_labels[indices[:-n_training_samples]]
    testset_data = iris_data[indices[-n_training_samples:]]
    testset_labels = iris_labels[indices[-n_training_samples:]]

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot training set points with different markers and labels
    colours = ("red", "green", "#bfa900")
    markers = ("_", "+", "v")
    labels = ("iris-setosa", "iris-versicolor", "iris-virginica")
    for iclass in range(3):
        ax.scatter(learnset_data[learnset_labels == iclass][:, 1],
                   learnset_data[learnset_labels == iclass][:, 2],
                   learnset_data[learnset_labels == iclass][:, 3],
                   c=colours[iclass], label=labels[iclass], marker=markers[iclass])

    # Plot test set points with different markers and labels
    test_colors = ("magenta", "blue", "cyan")
    test_markers = "*"
    test_labels = ("test Iris-setosa", "test Iris-versicolor", "test Iris-virginica")
    for i in range(3):
        ax.scatter(testset_data[testset_labels == i][:, 1],
                   testset_data[testset_labels == i][:, 2],
                   testset_data[testset_labels == i][:, 3],
                   c=test_colors[i], marker=test_markers, label=test_labels[i])

    # Set plot title and labels
    plt.title("KNN Classification - Iris Data and Test Points. (123)")
    ax.set_xlabel("Sepal Length")
    ax.set_ylabel("Sepal Width")
    ax.set_zlabel("Petal Length")

    # Add legend
    ax.legend(bbox_to_anchor=(0.20, 1))
    plt.show()

    # Create a KNN classifier
    knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
                               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                               weights='uniform')

    # Fit the classifier to the training data
    knn.fit(learnset_data, learnset_labels)

    # Display classifier parameters
    print("Number of test cases: " + str(n_training_samples))
    print("k = 5")
    print("KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',\n"
          "metric_params=None, n_jobs=1, n_neighbors=5, p=2, weights='uniform')")

    # Display true labels and predicted labels for the test set
    print("\nTarget values:")
    print(testset_labels)
    print("Predictions from the classifier:")
    print(knn.predict(testset_data))


def perceptron():
    # Load Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    # Split the data into 70% training data and 30% test data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    # Standardize the features
    sc = StandardScaler()
    sc.fit(X_train)

    # Apply the scaler to the X training and test data
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    ppn = Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,
                     fit_intercept=True, max_iter=1000, n_iter_no_change=5, n_jobs=None,
                     penalty=None, random_state=0, shuffle=True, tol=0.001,
                     validation_fraction=0.1, verbose=0, warm_start=False)

    print("\nPerceptron Model:")
    print(f"Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,\n"
          "fit_intercept=True, max_iter=1000, n_iter_no_change=5, n_jobs=None,\n"
          "penalty=None, random_state=0, shuffle=True, tol=0.001,\n"
          "validation_fraction=0.1, verbose=0, warm_start=False")

    # Train the perceptron
    ppn.fit(X_train_std, y_train)

    # Apply the trained perceptron on the X data to make predictions for the y test data
    y_pred = ppn.predict(X_test_std)

    # Display the predicted and true y test data
    print("\nTest sample labels:")
    print(y_test)
    print("Test samples classified as:")
    print(y_pred)

    # Calculate and display the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print("\nAccuracy: %.2f" % accuracy)

    # Create and display the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Plot the confusion matrix
    plt.matshow(conf_matrix, cmap='Blues')
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[i])):
            plt.text(j, i, str(conf_matrix[i][j]), ha='center', va='center',
                     color='black', fontsize=12)

    plt.show()


def binary_perceptron():
    # Load Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    # # Create a binary classification dataset for classes 1 and 2 (rows 0-100)
    # X_binary = X[:100]
    # Y_binary = (Y[:100] == 1).astype(int)  # Classes 1 and 2 are mapped to 0, 1

    # Create a binary classification dataset for classes 2 and 3 (rows 50-150)
    X_binary = X[50:150]
    Y_binary = (Y[50:150] == 2).astype(int) + 1  # Classes 2 and 3 are mapped to 0, 1

    # Split the binary dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_binary, Y_binary, test_size=0.2)

    # Standardize the features
    sc = StandardScaler()
    sc.fit(X_train)

    # Apply the scaler to the X training and test data
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Create a perceptron object with specified parameters
    ppn = Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,
                     fit_intercept=True, max_iter=1000, n_iter_no_change=5, n_jobs=None,
                     penalty=None, random_state=0, shuffle=True, tol=0.001,
                     validation_fraction=0.1, verbose=0, warm_start=False)

    # Display the Perceptron parameters
    print("\nPerceptron Model:")
    print(f"Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,\n"
          "fit_intercept=True, max_iter=1000, n_iter_no_change=5, n_jobs=None,\n"
          "penalty=None, random_state=0, shuffle=True, tol=0.001,\n"
          "validation_fraction=0.1, verbose=0, warm_start=False)\n")

    # Train the perceptron
    ppn.fit(X_train_std, y_train)

    # Apply the trained perceptron on the X data to make predictions for the y test data
    y_pred = ppn.predict(X_test_std)

    # Display the predicted and true y test data
    print("Test sample labels:")
    print(y_test)
    print("Test samples classified as:")
    print(y_pred)

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.2f}")

    # Print the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix:")
    print(conf_matrix)

    # Calculate and print precision, recall, f1, and ROC_AUC scores
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    roc_auc = roc_auc_score(y_test, y_pred, average=None)

    print("\nPrecision =", precision)
    print("Recall =", recall)
    print("F1 =", f1)
    print("ROC AUC =", roc_auc)

    # Plot the confusion matrix
    plt.matshow(conf_matrix, cmap='Blues')

    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Add numbers inside each box
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[i])):
            plt.text(j, i, str(conf_matrix[i][j]), ha='center', va='center',
                     color='black', fontsize=12)

    plt.show()


def main():
    knn_classification()
    perceptron()
    binary_perceptron()


if __name__ == '__main__':
    main()

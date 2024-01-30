from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


def display_digits(images, cmap='gray'):
    plt.gray()
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i], cmap=cmap)
        plt.axis('off')
    plt.show()


def main():
    digits = load_digits()
    print(f'Number of rows and columns of the dataset: {digits.data.shape}\n')
    # display_digits(digits.images)

    print(digits.target)

    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    mlp = MLPClassifier(
        activation='logistic',
        hidden_layer_sizes=[30, 10, 20],
        alpha=0.0001,
        learning_rate='invscaling',
        max_iter=666
    )

    print(mlp)  # print the parameters of the MLP classifier
    mlp.fit(X_train_std, y_train)
    y_pred = mlp.predict(X_test_std)

    print('\nTest samples labels:')
    print(y_test)
    print('\nTest samples classified as predicted')
    print(y_pred)

    confusion_matrix1 = confusion_matrix(y_test, y_pred)
    print(f'\nconfusion_matrix:')
    print(confusion_matrix1)
    print(f'\nAccuracy: {accuracy_score(y_test, y_pred):.2f}')
    print(f'Training set score: {mlp.score(X_train_std, y_train):.2f}')
    print(f'Test set score: {mlp.score(X_test_std, y_test):.2f}')


if __name__ == "__main__":
    main()


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split


def sheet_verify() -> int:
    while True:
        try:
            user_sheet = int(input('Enter the sheet number (range 1 - 4) of the '
                                   'dataset to be analyzed: '))
            if 1 <= user_sheet < 5:
                break
            else:
                raise ValueError
        except ValueError:
            print('Sheet number must be within the range of 1 - 4.')

    return user_sheet


def prediction():
    try:
        # Obtain the sheet number for analysis.
        sheet_number = sheet_verify()

        # Customize the number of columns used for training.
        columns_to_use = [0]

        # Adjust the test size for various test runs as needed.
        test_size = 0.30

        df = pd.read_excel('iris.xlsx', sheet_name=f'Sheet{sheet_number}')
        num_rows = len(df)

        print(f"Length(rows) of dataset: {num_rows}")
        print(f"Columns used in training dataset (explanatory variables): {columns_to_use}")
        print("Target estimation (response variable) data in column: 3 - The 'Petal width'")
        print(f"Test size used in the dataset: {test_size:.2f}")

        x = np.array(df[df.columns[0:2]])
        y = np.array(df[df.columns[3]])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

        model = LinearRegression()
        model.fit(x_train, y_train)
        y_predictions = model.predict(x_test)

        print("y_test\t\ty_prediction")
        for i in range(len(y_test)):
            print(f"{i} {y_test[i]:.2f}\t\t{y_predictions[i]:.2f}")

        r_squared = model.score(x_test, y_test)
        print(f"\nR-squared: {r_squared:.4f}\n")

    except Exception as e:
        print(e)


def classification():
    try:
        # Adjust the test size for various test runs as needed.
        test_size = 0.14
        df = pd.read_excel('iris.xlsx', sheet_name='Sheet1')
        num_rows = len(df)

        print(f"Length(rows) of dataset: {num_rows}")
        print("Columns used as features (explanatory variables): 0 - 3")
        print("Target (label) column: 4 - The 'Species'")
        print(f"Test size used in the dataset: {test_size:.2f}")

        x = np.array(df[df.columns[0:3]])
        y = np.array(df[df.columns[4]])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

        lr = LogisticRegression(max_iter=100)
        lr.fit(x_train, y_train)
        y_predictions = lr.predict(x_test)

        print(lr)
        print('y_test\t\t\ty_prediction')

        for i in range(0, len(y_predictions), 1):
            print(f'{i} {y_test[i]}\t\t{y_predictions[i]}')

        r_squared = lr.score(x_test, y_test)
        print(f'\nR-squared: {r_squared:.4f}')
    except Exception as e:
        print(e)


def main():
    prediction()
    classification()


if __name__ == '__main__':
    main()

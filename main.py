from preprocessing.preprocess_data import preprocess_file
from models.random_forest import main_rf
from models.logistic_regression import main_lr


def main():
    preprocess_file()
    main_rf()
    main_lr()


if __name__ == '__main__':
    main()

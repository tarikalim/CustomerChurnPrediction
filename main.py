from preprocessing.preprocess_data import preprocess_file
from models.random_forest import main_rf


def main():
    preprocess_file()
    main_rf()


if __name__ == '__main__':
    main()

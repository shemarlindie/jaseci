from .train import load_model
from .config import test_file_path
from .predict import predict

if __name__ == '__main__':
    model = load_model()

    print('')
    print('TESTING')
    with open(test_file_path, 'r', encoding='utf-8') as input_file:
        tests = [ln.strip() for ln in input_file.readlines() if ln.strip()]

    result = predict(tests, model)
    print(result)
    print('-' * 20)

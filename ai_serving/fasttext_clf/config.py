import os


def make_path(file):
    return os.path.join(os.path.dirname(__file__), file)


model_file_path = make_path('model.ftz')
train_file_path = make_path('train.txt')
test_file_path = make_path('test.txt')
clf_json_file_path = make_path('training_data.json')

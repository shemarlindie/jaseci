import sys

from .train import load_model
from .json_to_train import label_to_intent, prep_sentence


def predict(sentences, model=None):
    model = model if model else load_model()
    result = {}
    for sentence in sentences:
        result[sentence] = []
        predictions = model.predict(prep_sentence(sentence))
        for pre in zip(predictions[0], predictions[1]):
            result[sentence].append({
                'sentence': sentence,
                'intent': label_to_intent(pre[0]),
                'probability': pre[1]
            })
    return result


if __name__ == '__main__':
    sent = sys.argv[1] if len(sys.argv) > 1 else exit(1)
    res = predict([sent])
    print('')
    print(f'Q: {sent}')
    print(res)

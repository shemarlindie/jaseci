from .ai_serving_api import AIServingAPI

FASTTEXT_CLF_API = AIServingAPI('FASTTEXT_CLF')


def predict(param_list, meta=None):
    """
    Predict intent
    Param 1 - list of strings (sentences)
    """
    data = {
        'op': 'predict',
        'sentences': param_list[0]
    }
    return FASTTEXT_CLF_API.post(data)

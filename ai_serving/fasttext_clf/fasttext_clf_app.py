from flask import request

from base.ai_serving_base import AIServiceBase, FlaskError

from .train import load_model
from .predict import predict


class FasttextClfBase:
    model = None

    def predict(self, sentences):
        self.model = self.model or load_model()
        return predict(sentences, self.model)


fasttext = FasttextClfBase()
FasttextClfApp = AIServiceBase(name='fasttext-clf')


@FasttextClfApp.app.route('/fasttext-clf/', methods=['POST'])
def __router__():
    print(request.json)
    if 'op' not in request.json:
        raise FlaskError(
                message='Required param \'op\' is missing from the request.',
                status_code=404
        )
    op = request.json['op']
    if op == 'predict':
        if 'sentences' in request.json:
            return fasttext.predict(request.json['sentences'])
        else:
            raise FlaskError(
                message='"sentences" not found in request',
                status_code=400,
                payload=request.json
            )

    raise FlaskError(
            message='Invalid param op. ' +
                    'Supported: ecnode',
            status_code=404, payload={'original_op': request.json['op']}
    )


if __name__ == '__main__':
    print('fasttext-clf service up and running')
    FasttextClfApp.run(port=4675)

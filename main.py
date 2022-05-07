from flask import Flask, request

from typing import Tuple
from typing import List

from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
import torch
import numpy as np


def load_model(model_path: str) -> Tuple[AutoTokenizer, AutoModelForTokenClassification]:
    model_file = model_path + "/" + "pytorch_model.bin"
    config = AutoConfig.from_pretrained(model_path + "/" + 'config.json')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cpu")
    state = torch.load(model_file, map_location=device)
    model = AutoModelForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=None, state_dict=state, config=config)
    return model, tokenizer


def convert_tokens_to_words(tokens: List[str], label_indices: List[int]) -> Tuple[List[str], List[str]]:
    index_2_label_map = {0: 'B', 1: 'I', 2: 'O', -100: 'X'}
    words, labels = [], []
    for token, label_index in zip(tokens, label_indices):
        if token.startswith("##"):
            words[-1] = words[-1] + token[2:]
        else:
            labels.append(index_2_label_map.get(label_index))
            words.append(token)
    return words, labels


def build_json_response(text: str, words: List[str], labels: List[str]) -> dict:
    tokens_with_start_end = []
    offset = 0
    for token, label in zip(words, labels):
        start = text.find(token, offset)
        if start != -1:
            offset = start + len(token)
        tokens_with_start_end.append((token, label, start, offset))
    ret = []
    for element in tokens_with_start_end:
        if element[1] == "O":
            if len(ret) > 0:
                ret[-1][0] = text[ret[-1][1]:ret[-1][2]]
        if element[1] == "B":
            ret.append(["", element[2], element[3]])
        if element[1] == "I":
            ret[-1][2] = element[3]
    response = {"text": text, "diseases": []}
    for element in ret:
        response["diseases"].append({"disease": element[0], "start": element[1], "end": element[2]})
    return response


def find_diseases(text: str, tokenizer: AutoTokenizer,
                  model: AutoModelForTokenClassification) -> dict:
    device = torch.device("cpu")
    sentence_tokens_as_ids = torch.tensor([tokenizer.encode(text)]).to(device)
    model.eval()
    with torch.no_grad():
        output = model(sentence_tokens_as_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
    tokens = tokenizer.convert_ids_to_tokens(sentence_tokens_as_ids.to('cpu').numpy()[0])
    words, labels = convert_tokens_to_words(tokens, label_indices[0])
    return build_json_response(text, words, labels)


class Context:
    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer

    def get_model(self):
        return self._model

    def get_tokenizer(self):
        return self._tokenizer

    def set_model(self, model):
        self._model = model

    def set_tokenizer(self, tokenizer):
        self._tokenizer = tokenizer


app = Flask(__name__)
context = Context(None, None)


@app.before_first_request
def before_first_request():
    model, tokenizer = load_model("DiseaseBert")
    context.set_model(model)
    context.set_tokenizer(tokenizer)
    print("model and tokenizer initialized")


@app.route("/annotate-diseases")
def annotate_diseases():
    text = request.args.get('text')
    if not text:
        return 'Please add the text request parameter like the following: <a ' \
               'href="http://127.0.0.1:8080/annotate-diseases?text=blablabla">http://127.0.0.1:8080/annotate-diseases' \
               '?text=blablabla</a> '
    return find_diseases(text, context.get_tokenizer(), context.get_model())


@app.route("/")
def root():
    return 'please type the following url to get some answer <a ' \
           'href="http://127.0.0.1:8080/annotate-diseases?text=blablabla">http://127.0.0.1:8080/annotate-diseases' \
           '?text=blablabla</a> '


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)

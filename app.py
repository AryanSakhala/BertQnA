import torch
from model import BertQnA
from flask import request
import flask
import os
from flask import Flask, render_template, request


# Tokenizer / model
from transformers import DistilBertForQuestionAnswering

model = DistilBertForQuestionAnswering.from_pretrained("model/")
# Tokenizer
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("model/")
import os
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import model
import torch
from transformers import BertForQuestionAnswering
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", pred="Please ask a question!")


@app.route("/predict", methods=["POST"])
def predict():
    data = [request.form["name"]]

    name = [request.form["question"]]

    answer = model.BertQnA(data[0], name[0])

    return render_template(
        "index.html", pred=f"{name} ...I think the answer is {answer} !?"
    )


if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

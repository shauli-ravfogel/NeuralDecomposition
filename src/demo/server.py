#!/usr/bin/python

import logging
from datetime import datetime

from flask import Flask, request
from flask_cors import CORS, cross_origin

import spacy

import sys
sys.path.append('../analysis/evaluate')
from evaluate import get_closest_sentence_demo, get_sentence_representations


app = Flask(__name__)
CORS(app)

nlp = spacy.load('en_core_web_sm')

with open(, "rb") as f:
    data = pickle.load(f)
sentence_reprs = get_sentence_representations(embds_and_sents)


def get_logger(model_dir):
    time = str(datetime.now()).replace(' ', '-')
    logger = logging.getLogger(time)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
    # create file handler which logs even debug messages
    fh = logging.FileHandler(model_dir + '/' + time + '.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = get_logger('./logs/')


def get_token_for_char(doc, char_idx):
    """
    Convert between the characted index to the nlp token index
    :param doc:
    :param char_idx:
    :return:
    """
    for i, token in enumerate(doc):
        if char_idx > token.idx:
            continue
        if char_idx == token.idx:
            return i
        if char_idx < token.idx:
            return i - 1


def get_nearest(text):
    text_split = text.split('*')
    ind = len(text_split[0]) + 1

    doc = nlp(''.join(text_split))
    token_ind = get_token_for_char(doc, ind)

    get_closest_sentence_demo()
    ans = [doc.text + ' test1', doc.text + ' test2']
    return '\n'.join(ans)


@app.route('/syntax_extractor/', methods=['GET'])
@cross_origin()
def serve():
    text = request.args.get('text')
    logger.info('request: ' + text)

    if text.strip() == '':
        return ''

    try:
        doc = nlp(text)

        nearest = get_nearest(doc)

        logger.info('ans: ' + str(nearest))
        html = nearest

    except Exception as e:
        logger.info('error. ' + str(e))
        html = 'some error occurred while trying to find the NFH'

    return html


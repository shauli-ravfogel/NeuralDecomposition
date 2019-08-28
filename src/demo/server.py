#!/usr/bin/python

import logging
from datetime import datetime

from flask import Flask, request
from flask_cors import CORS, cross_origin

import spacy


app = Flask(__name__)
CORS(app)

nlp = spacy.load('en_core_web_sm')


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


def get_nearest(doc):
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


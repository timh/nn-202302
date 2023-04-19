import json
from typing import List, Dict, Tuple
from flask import Flask, request, make_response, Response
import datetime
from pathlib import Path

import sys
sys.path.append("..")
sys.path.append("../..")

from experiment import Experiment
import checkpoint_util

app = Flask(__name__)

def make_error(msg: str, code: int) -> Response:
    error = {
        "message": msg
    }
    resp = make_response(json.dumps(error, indent=2), code)
    resp.headers["Content-Type"] = "application/json"
    return resp

def make_json(obj) -> Response:
    if isinstance(obj, Experiment):
        obj = obj.to_dict()
    resp = make_response(json.dumps(obj, indent=2))
    resp.headers["Content-Type"] = "application/json"
    return resp

@app.route('/nn-api/experiments')
def list_experiments():
    exps = checkpoint_util.list_experiments(runs_dir=Path("../runs"))
    res = [exp.metadata_dict() for exp in exps]
    return make_json(res)

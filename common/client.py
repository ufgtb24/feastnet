# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A client that performs inferences on a ResNet model using the REST API.

The client downloads a test image of a cat, queries the server over the REST API
with the test image repeatedly and measures how long it takes to respond.

The client expects a TensorFlow Serving ModelServer running a ResNet SavedModel
from:

https://github.com/tensorflow/models/tree/master/official/resnet#pre-trained-model

The SavedModel must be one that can take JPEG images as inputs.

Typical usage example:

    resnet_client.py
"""

from __future__ import print_function

import json

from Direction.src.dire_data import process_data

import requests

# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
SERVER_URL = 'http://192.168.1.119:8501/v1/models/direction:predict'


def build_instance(vertice, adjs, perms):
    feed_dict = {
        'vertice': vertice.tolist(),
    }
    
    adjs_dict = {'adj_%d' % i: adj.tolist() for i, adj in enumerate(adjs)}
    perms_dict = {'perm_%d' % i: perm.tolist() for i, perm in enumerate(perms)}
    feed_dict.update(adjs_dict)
    feed_dict.update(perms_dict)
    instance=json.dumps({"instances":[feed_dict], "signature_name": "serving_default"})
    return instance


def main():
    data_path = "F:/ProjectData/mesh_direction/2aitest/low"
    X, Adjs, Perms = process_data(data_path, 'case_test.txt')
    x, adjs, perms =X[0], Adjs[0], Perms[0]

    predict_request = build_instance(x,adjs,perms)
    print('predict_request:   ',predict_request)
    headers = {"content-type": "application/json"}
    # Send few requests to warm-up the model.
    for _ in range(3):
        response = requests.post(SERVER_URL, data=predict_request, headers=headers)
        print('r.txt:  ',response.text)
        response.raise_for_status()
        print('r.json:  ',response.json)

        # Send few actual requests and report average latency.
    total_time = 0
    num_requests = 10
    for _ in range(num_requests):
        response = requests.post(SERVER_URL, data=predict_request)
        response.raise_for_status()
        total_time += response.elapsed.total_seconds()
        prediction = response.json()['predictions'][0]
        
    print('Prediction class: {}, avg latency: {} ms'.format(
      prediction['output_node'], (total_time*1000)/num_requests))


if __name__ == '__main__':
    main()

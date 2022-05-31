import os
import argparse
import json
import pickle
import sys
import torch
from torch import nn, optim
from torch.autograd import Variable
import boto3
import logging


# Initialised AWS CloudWatch Logger
logger = logging.getLogger()

# Initialised and Authenticated AWS S3 Service
bucketname = "sagemaker-fad"
itemname = "model.pth"
s3 = boto3.resource('s3',
                    aws_access_key_id="-",
                    aws_secret_access_key= "-")
# Downloaded model
obj = s3.Object(bucketname, itemname).download_file('model.pth')

# Inference Script functions:
def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading the model.')
    model = nn.Sequential(nn.Linear(8, 128),
                          nn.ReLU(),
                          nn.Linear(128, 512),
                          nn.ReLU(),
                          nn.Linear(512, 1024),
                          nn.ReLU(),
                          nn.Linear(1024, 256),
                          nn.ReLU(),
                          nn.Linear(256, 2),)

    with open('model.pth', 'rb') as f:
        model.load_state_dict(torch.load(f))
        
    model.to(device).eval()
    logger.info('Done loading model')
    return model

def input_fn(request_body, content_type='application/json'):
    logger.info('Deserializing the input data.')
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        data = input_data["input"]
        logger.info(f'List: {data}')
        
        return data
    raise Exception(f'Requested unsupported ContentType in content_type {content_type}')
    
def predict_fn(input_data, model):
    logger.info('Generating prediction based on input parameters.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = Variable(torch.Tensor(input_data).float())
    data = data.to(device)

    model.eval()

    with torch.no_grad():
        output = model(data)
    result = output.argmax().item()
    return result

def output_fn(prediction, accept='application/json'):
    logger.info('Serializing the generated output.')
    result = prediction
    if accept == 'application/json':
        return json.dumps(result), accept
  

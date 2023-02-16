# deploy a model using tf serving and stress test it.

This tutorial is based on this medium post https://towardsdatascience.com/use-pre-trained-huggingface-models-in-tensorflow-serving-d2761f7e69f6 which shows how to use huggingface models on tf serving. 

I decided to do additional steps apart of the one shown there. I created a golang client to call the model inference and I also stress tested the model on different machines.

This tutorial has the following sections:

1. Installation instructions
2. How to get a TF SavedModel
3. How to serve your model using TF Serving on Docker


# Installation instructions

You may have to install tensorflow in your local machine for running tests

## Mac and linux with amd architecture

If you are using a linux amd computer you can use conda and install tensorflow and transformers without any problems.

## Linux ARM

For mac m1, you have to use miniforge or miniconda. It took some minutes for me to figure out how to install it, you can check the following resources:

https://developer.apple.com/metal/tensorflow-plugin/
https://developer.apple.com/forums/thread/702851
https://jamescalam.medium.com/hugging-face-and-sentence-transformers-on-m1-macs-4b12e40c21ce

What worked for me was to install miniforge with the first link and then I ran the following, make sure that tensorflow-deps and tensorflow macos have the same version :

``` bash
conda install -c apple tensorflow-deps
python -m pip install tensorflow-macos==2.9
python -m pip install tensorflow-metal==0.5 
```

# Obtain tensorflow saved model.

You can deploy on tensorflow serving by first obtaining a [SavedModel](https://www.tensorflow.org/guide/saved_model), which is a complete tf program, including tf variables and computation, so that you can easily deploy it with tflite, tf.js or tf serving.


## Model selection, Sentiment analysis - HuggingFace

I went to check the hugging face models to look for something related with sentiment analysis and went for a simple bert model tuned for sentiment analysis https://huggingface.co/textattack/bert-base-uncased-SST-2. It has .pth checkpoints for pytorch. I selected a simple bert, because I'll be running inference from different clients, including golang and python backends, and I was afraid about not being able to use the huggingface tokenizer on Golang.

## Saving a hugging face pytorch model as tf model

As there are no tensorflow checkpoints for this model [here](https://huggingface.co/textattack/bert-base-uncased-SST-2/tree/main), 
you can save the pytorch model in tensorflow format by running the [convert script](./hugging_face/convert_pytorch_to_tf.py). Which loads the transformer model from the pytorch weights and then save it was if it were a tensorflow model.

```bash
python hugging_face/convert_pytorch_to_tf.py 
```

# Deploy your model 

In order to deploy your model, the recommended way is to use docker to run tf serving. My preferred way is by creating [my own serving image](https://www.tensorflow.org/tfx/serving/docker#creating_your_own_serving_image):


Remember that now is better to keep using amd instead of arm, so move to a linux instance with amd if necessary.

follow steps here: https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/docker.md#creating-your-own-serving-image

You copy the model and commit 
- docker run -d --name serving_base tensorflow/serving
- docker cp <path to your saved model folder>/bert-base-uncased-SST-2 servig_base
- docker commit --change "ENV MODEL_NAME bert-base-uncased-SST-2" serving_base bert-base-uncased-SST-2-image


Run a container with your image
- docker run -p 8501:8501  -t bert-base-uncased-sst2-image


# stress test

After installing locust you can run it with the following cmd:

locust --host=http://ec2-3-132-201-36.us-east-2.compute.amazonaws.com:8501 -f inference_clients/locust_client.py



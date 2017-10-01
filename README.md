# Udacity Robotics Project 4 - Follow Me

In this project, we have to build and train a Fully Convolutional Network using Keras.
The Network is trained to identify a target person (red hair, red shirt) from an image.

In the Quad-Copter Simulator, the Quad-Copter first move in patroling mode, and keeps sending camera images to our controller script.
The window on the right shows 2 videos side by side. The left video is the camera feed, the right video is the analysis done by the Fully Convolutional Network (Blue: target, Green: other people, Pink: not people)
Once the controller identify the target (at 1:07), the controller script instructs the Quad-Copter to follow the target by keeping the Blue region in the middle of the camera view.


The project model training notebook files are:
1. model_training.ipynb
2. model_training.html

## Model Training: Network Architecture

A fully convolutional network (FCN) is used to traing the semantic segmentation model.

The FCN contains 3 encoder blocks, followed by a 1x1 convolution layer, and 3 sysmetrical decoder blocks.

```
def fcn_model(inputs, num_classes):

    # Encoder Blocks.
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    enc1 = encoder_block(inputs, 64, 2)
    enc2 = encoder_block(enc1, 128, 2)
    enc3 = encoder_block(enc2, 256, 2)
    # 1x1 Convolution layer using conv2d_batchnorm().
    onexone = conv2d_batchnorm(enc3, 256, kernel_size=1, strides=1)
    # Decoder Blocks
    dec3 = decoder_block(onexone, enc2, 256)
    dec2 = decoder_block(dec3, enc1, 128)
    dec1 = decoder_block(dec2, inputs, 64)

    return layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(dec1)
```

Each encoder block contains 1 separable convolution layer.

```
def encoder_block(input_layer, filters, strides):

    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer
```

Each decoder block has 1 bilinear upsampling layer, a layer concatenation step, and 1 separable convolution layers. The separable convolution layers helps to extract more spatial information from the previous layers.

```
def decoder_block(small_ip_layer, large_ip_layer, filters):

    # TODO Upsample the small input layer using the bilinear_upsample() function.
    upsample_layer = bilinear_upsample(small_ip_layer)
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    cat_layer = layers.concatenate([upsample_layer, large_ip_layer])
    # TODO Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(cat_layer, filters, 1)
    return output_layer
```

The decoder includes a concatenation step which concatenates the current input with the output of a layer a few steps before it. This concatenation helps the network to retain more spatial information that were "lost" in the in between layers

Since the decoder contains a concatenation step, the size of the output of the encoders and decoders need to be the same so that there are equal number of parameters to concatenate. That is why the strides of the encoders are all set to 2, and the bilinear upsampling in the decoder is set to 2 also.

The number of filters should increase with each layer of the encoder and decrease with each layer of decoder. The result seems to get better when I increase the  number of filters. But I could not increase it beyond 64 due to the limitation of the 8GB Ram of the Nvidia GTX 1070 GPU that I am using.


## Model Training: Hyperparameters

The following Hyperparameters were used for the final run to train the FCN.

```
learning_rate = 0.01
batch_size = 64
num_epochs = 10
steps_per_epoch = 200
validation_steps = 50
workers = 2
```

I started with learning_rate = 0.01, and didn't change it because the network seems to be training well with the error reducing constantly.

I have to keep the batch_size at 64 due to the limitation of the 8GB Ram of the Nvidia GTX 1070 GPU that I am using.

The value of steps_per_epoch and validation_steps depends on the number of training and validation images available. For the amount of training and validation images I have, I guess 200 and 50 are reasonable numbers.

During training, the GPU utilization was at 100%, which is good. So I kept the number of workers at 2.

## Model Training: 1x1 Convolution Layer vs Fully Connected Layer

In a traditional Convolution Network, the encoder layers are usually followed by 1 or more fully connected layers. The fully connected layers helps the network to "ignore" spatial information, and is useful in identifying objects regardless of where the objects are located in the image.

In the Fully Convolution Network, we not only want to identifying whether the objects are in the image, we also want to know where the objects are.

So we replace the fully connected layers with 1x1 Convolution Layer, which helps to retain the spatial information from the input image.

## Model Training: Encoding and Decoding

In the Fully Convolution Network, the encoding layers uses convolution to help it identifying objects regardless of where the objects are located in the image.

The decoding layers helps to identify the location of the identified objects, down to the pixel level.

Each decoder layer make use of the skip connection technique by concatenating the current input with the output of a layer a few steps before it. This concatenation helps the network to retain more spatial information that were "lost" in the in between layers.



## Model Training: Limitations

This 3 layers FCN has enough capacity to identify and locate objects in fairly complex scenes.

It will work well in other scenarios as long as we have enough training data with the target objects clearly labeled.

In this project, the hero is distinctly different from the surroundings. She has red hair and shirt, while the surrounding background don't have red objects. Even for the crowd, there is only 1 other woman model with red pants which is more likely to cause false identifications. This makes the training of the FCN fairly easy.

The performance of the network may not be as good if the object (dog, cat, car, etc.) is not distinctly different from the surroundings.

More training data and a more complex FCN architecture will be needed for real life scenarios.

## Model Training: Results

The training of the FCN is done using a Nvidia GTX 1070 GPU.

![Model Training](https://github.com/ongchinkiat/robond-follow-me/raw/master/fcn-training-curve.jpg "Model Training")

Training loss decreases consistently, but validation loss fluctuates around 0.04 after only 6 epochs. This may be a sign of over fitting.

Also, I encountered out of memory errors using the same network, at 12 epochs. I guess the network is using very near to the max memory of the graphics card, so every slight increase of memory usage after each epoch (possibly some memory leak) increases the risk of memory error.

Thus I limit the training to 10 epochs.

The model files are
1. model_weights_given_set.h5
2. config_model_weights_given_set.h5

The final Intersection over Union (IoU) for the network is 54.5% (0.545).


## Simulation

The model trained is run to control the Quad-Ropter in the Unity Simulator.

```
python follower.py model_weights_given_set.h5
```

The model seems to work well in the simulator, identifying the hero on first encounter every time, and has no problem following the hero even if she is making shape turns.

I have uploaded a video of a full simulation run of the Quad-Copter Simulator with the python controller script controlling the Quad-Copter to identify and follow the target.

Video URL: https://youtu.be/I8zjX-zyWBs

<a href="http://www.youtube.com/watch?feature=player_embedded&v=I8zjX-zyWBs" target="_blank"><img src="http://img.youtube.com/vi/I8zjX-zyWBs/0.jpg"
alt="Follow Me" width="240" height="180" border="1" /></a>

## Future Enhancements

Base on the observations made while tuning the hyperparameters and network architecture, the performance of the FCN can still experience much improvements by simply having more layers and larger filter sizes.

Doing these at an acceptable training speed would require the use of more powerful graphics card with more RAM, then the GTX 1070 with 8GB RAM that I am using.

Also more data images could be added to the training and validation data sets, to improve the learning and reduce the risk of over fitting.

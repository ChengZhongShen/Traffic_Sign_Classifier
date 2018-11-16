import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

def LeNet(x, keep_prob_conv, keep_prob_fc):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Change RGB to Gray
    x = tf.image.rgb_to_grayscale(x)     
    
    # normalize the data
    x = tf.map_fn(lambda image: tf.image.per_image_standardization(image), x)    
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    weight_c1 = tf.Variable(tf.truncated_normal([5,5,1,6], mean=mu, stddev=sigma))     
    biases_c1 = tf.Variable(tf.zeros([6]))
    conv1 = tf.nn.conv2d(x, weight_c1, strides=[1,1,1,1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, biases_c1)
    # Activation.
    conv1 = tf.nn.relu(conv1)
    # Dropout
    conv1 = tf.nn.dropout(conv1, keep_prob_conv)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    # Layer 2: Convolutional. Output = 10x10x16.
    weight_c2 = tf.Variable(tf.truncated_normal([5,5,6,16], mean=mu, stddev=sigma))
    biases_c2 = tf.Variable(tf.zeros([16]))
    conv2 = tf.nn.conv2d(conv1, weight_c2, strides=[1,1,1,1], padding='VALID')
    conv2 = tf.nn.bias_add(conv2, biases_c2)
    # Activation.
    conv2 = tf.nn.relu(conv2)
    # Dropout
    conv2 = tf.nn.dropout(conv2, keep_prob_conv)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    # Flatten. Input = 5x5x16. Output = 400.
    conv2_flat = flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    weights_3 = tf.Variable(tf.truncated_normal([400, 120], mean=mu, stddev=sigma))
    biases_3 = tf.Variable(tf.zeros([120]))
    fc1 = tf.add(tf.matmul(conv2_flat, weights_3), biases_3)
    # Activation.
    fc1 = tf.nn.relu(fc1)
    # Dropout
    fc1 = tf.nn.dropout(fc1, keep_prob_fc)
    
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    weights_4 = tf.Variable(tf.truncated_normal([120, 84], mean=mu, stddev=sigma))
    biases_4 = tf.Variable(tf.zeros([84]))
    fc2 = tf.add(tf.matmul(fc1, weights_4), biases_4)
    # Activation.
    fc2 = tf.nn.relu(fc2)
    # Dropout
    fc2 = tf.nn.dropout(fc2, keep_prob_fc)
    
    # Layer 5: Fully Connected. Input = 84. Output = 43.
    weights_5 = tf.Variable(tf.truncated_normal([84,43], mean=mu, stddev=sigma))
    biases_5 = tf.Variable(tf.zeros([43]))
    logits = tf.add(tf.matmul(fc2, weights_5), biases_5)
    return logits

def LeNet_4x(x, keep_prob_conv, keep_prob_fc):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Change RGB to Gray
    x = tf.image.rgb_to_grayscale(x) 
    
    # normalize the data
    x = tf.map_fn(lambda image: tf.image.per_image_standardization(image), x)
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x24.
    weight_c1 = tf.Variable(tf.truncated_normal([5,5,1,24], mean=mu, stddev=sigma))
    biases_c1 = tf.Variable(tf.zeros([24]))
    conv1 = tf.nn.conv2d(x, weight_c1, strides=[1,1,1,1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, biases_c1)
    # Activation.
    conv1 = tf.nn.relu(conv1)
    # Dropout
    conv1 = tf.nn.dropout(conv1, keep_prob_conv)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    # Layer 2: Convolutional. Output = 10x10x16.
    weight_c2 = tf.Variable(tf.truncated_normal([5,5,24,64], mean=mu, stddev=sigma))
    biases_c2 = tf.Variable(tf.zeros([64]))
    conv2 = tf.nn.conv2d(conv1, weight_c2, strides=[1,1,1,1], padding='VALID')
    conv2 = tf.nn.bias_add(conv2, biases_c2)
    # Activation.
    conv2 = tf.nn.relu(conv2)
    # Dropout
    conv2 = tf.nn.dropout(conv2, keep_prob_conv)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    # Flatten. Input = 5x5x16. Output = 400.
    conv2_flat = flatten(conv2)
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    weights_3 = tf.Variable(tf.truncated_normal([1600, 480], mean=mu, stddev=sigma))
    biases_3 = tf.Variable(tf.zeros([480]))
    fc1 = tf.add(tf.matmul(conv2_flat, weights_3), biases_3)
    # Activation.
    fc1 = tf.nn.relu(fc1)
    # Dropout
    fc1 = tf.nn.dropout(fc1, keep_prob_fc)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    weights_4 = tf.Variable(tf.truncated_normal([480, 336], mean=mu, stddev=sigma))
    biases_4 = tf.Variable(tf.zeros([336]))
    fc2 = tf.add(tf.matmul(fc1, weights_4), biases_4)
    # Activation.
    fc2 = tf.nn.relu(fc2)
    # Dropout
    fc2 = tf.nn.dropout(fc2, keep_prob_fc)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    weights_5 = tf.Variable(tf.truncated_normal([336,43], mean=mu, stddev=sigma))
    biases_5 = tf.Variable(tf.zeros([43]))
    logits = tf.add(tf.matmul(fc2, weights_5), biases_5)
    return logits

def LeNet_4x_MS(x, keep_prob_conv, keep_prob_fc):
    '''
    LeNet expand the hidden layers 4x and mulitle scaled(conv1 also connected to classifier, fc1)
    '''    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Change RGB to Gray
    x = tf.image.rgb_to_grayscale(x)

    # normalize the data
    x = tf.map_fn(lambda image: tf.image.per_image_standardization(image), x)
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x24.
    weight_c1 = tf.Variable(tf.truncated_normal([5,5,1,24], mean=mu, stddev=sigma))
    biases_c1 = tf.Variable(tf.zeros([24]))
    conv1 = tf.nn.conv2d(x, weight_c1, strides=[1,1,1,1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, biases_c1)
    # Activation.
    conv1 = tf.nn.relu(conv1)
    # Dropout
    conv1 = tf.nn.dropout(conv1, keep_prob_conv)
    # Pooling. Input = 28x28x24. Output = 14x14x24.
    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # flatten
    conv1_flat = flatten(conv1) # 14*14*24 = 4704
    
    # Layer 2: Convolutional. Input 14*14*24 Output = 10x10x64.
    weight_c2 = tf.Variable(tf.truncated_normal([5,5,24,64], mean=mu, stddev=sigma))
    biases_c2 = tf.Variable(tf.zeros([64]))
    conv2 = tf.nn.conv2d(conv1, weight_c2, strides=[1,1,1,1], padding='VALID')
    conv2 = tf.nn.bias_add(conv2, biases_c2)
    # Activation.
    conv2 = tf.nn.relu(conv2)
    # Pooling. Input = 10x10x64. Output = 5x5x64.
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # Flatten. Input = 5x5x64. Output = 1600.
    conv2_flat = flatten(conv2) # 
    
    # combine conv1/conv2
    conv_flat = tf.concat([conv1_flat, conv2_flat],1) #(6304=4704+1600)
        
    # Layer 3: Fully Connected. Input = 6304. Output = 480.
    weights_3 = tf.Variable(tf.truncated_normal([6304, 480], mean=mu, stddev=sigma))
    biases_3 = tf.Variable(tf.zeros([480]))
    fc1 = tf.add(tf.matmul(conv_flat, weights_3), biases_3)
    # Activation.
    fc1 = tf.nn.relu(fc1)
    # Dropout
    fc1 = tf.nn.dropout(fc1, keep_prob_fc)
    
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    weights_4 = tf.Variable(tf.truncated_normal([480, 336], mean=mu, stddev=sigma))
    biases_4 = tf.Variable(tf.zeros([336]))
    fc2 = tf.add(tf.matmul(fc1, weights_4), biases_4)
    # Activation.
    fc2 = tf.nn.relu(fc2)
    # Dropout
    fc2 = tf.nn.dropout(fc2, keep_prob_fc)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    weights_5 = tf.Variable(tf.truncated_normal([336,43], mean=mu, stddev=sigma))
    biases_5 = tf.Variable(tf.zeros([43]))
    logits = tf.add(tf.matmul(fc2, weights_5), biases_5)
    return logits

def network(x, keep_prob_conv, keep_prob_fc):
    """
    a net work according the below paper
    http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
    """    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Change RGB to Gray
    x = tf.image.rgb_to_grayscale(x)

    # normalize the data
    x = tf.map_fn(lambda image: tf.image.per_image_standardization(image), x)
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 32x32x30.
    weight_c1 = tf.Variable(tf.truncated_normal([5,5,1,30], mean=mu, stddev=sigma))     
    biases_c1 = tf.Variable(tf.zeros([30]))
    conv1 = tf.nn.conv2d(x, weight_c1, strides=[1,1,1,1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, biases_c1)
    # Activation.
    conv1 = tf.nn.relu(conv1)
    # Dropout
    conv1 = tf.nn.dropout(conv1, keep_prob_conv)
    # Pooling. Input = 32x32x30. Output = 16x16x30.
    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # Flatten. Input = 16x16x30. Output = 7680.
    conv1_flat = flatten(conv1)
    
    # Layer 2: Convolutional. Input 16X16X30, Output = 16x16x15.
    weight_c2 = tf.Variable(tf.truncated_normal([5,5,30,15], mean=mu, stddev=sigma))
    biases_c2 = tf.Variable(tf.zeros([15]))
    conv2 = tf.nn.conv2d(conv1, weight_c2, strides=[1,1,1,1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, biases_c2)
    # Activation.
    conv2 = tf.nn.relu(conv2)
    # Dropout
    conv2 = tf.nn.dropout(conv2, keep_prob_conv)
    # Pooling. Input = 16x16x15. Output = 8x8x15.
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # Flatten. Input = 8x8x64. Output = 960.
    conv2_flat = flatten(conv2)
       
    # Layer 3: Convolutional, Input=8x8x15, Output=8x8x10
    weight_c3 = tf.Variable(tf.truncated_normal([5,5,15,10], mean=mu, stddev=sigma))
    biases_c3 = tf.Variable(tf.zeros([10]))
    conv3 = tf.nn.conv2d(conv2, weight_c3, strides=[1,1,1,1], padding='SAME')
    conv3 = tf.nn.bias_add(conv3, biases_c3)
    # Activation
    conv3 = tf.nn.relu(conv3)
    # Dropout
    conv3 = tf.nn.dropout(conv3, keep_prob_conv)
    # Pooling. Input=8x8x10, Output=4x4x10
    conv3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # Flatten Input=4x4x10, output = 160
    conv3_flat = flatten(conv3)
    
    # combine conv1/conv2/conv3
    conv_flat = tf.concat([conv1_flat, conv2_flat, conv3_flat],1) #(8800=7680+960+160)
    
    # Layer 4: Fully Connected. Input = 8800. Output = 960.
    weights_3 = tf.Variable(tf.truncated_normal([8800, 960], mean=mu, stddev=sigma))
    biases_3 = tf.Variable(tf.zeros([960]))
    fc1 = tf.add(tf.matmul(conv_flat, weights_3), biases_3)
    # Activation.
    fc1 = tf.nn.relu(fc1)
    # Dropout
    fc1 = tf.nn.dropout(fc1, keep_prob_fc)
        
    # Layer 5: Fully Connected. Input = 960. Output = 336
    weights_4 = tf.Variable(tf.truncated_normal([960, 336], mean=mu, stddev=sigma))
    biases_4 = tf.Variable(tf.zeros([336]))
    fc2 = tf.add(tf.matmul(fc1, weights_4), biases_4)
    # Activation.
    fc2 = tf.nn.relu(fc2)
    # Dropout
    fc2 = tf.nn.dropout(fc2, keep_prob_fc)
    
    # Layer 6: Fully Connected. Input = 336 Output = 43.
    weights_5 = tf.Variable(tf.truncated_normal([336,43], mean=mu, stddev=sigma))
    biases_5 = tf.Variable(tf.zeros([43]))
    logits = tf.add(tf.matmul(fc2, weights_5), biases_5)
    return logits 
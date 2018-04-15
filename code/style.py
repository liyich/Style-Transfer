from __future__ import print_function
from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imsave
import numpy as np
from scipy.optimize import fmin_l_bfgs_b,fmin_ncg,fmin_tnc,fmin_cobyla,fmin_cg,fmin_bfgs
import time
import argparse
import matplotlib.pyplot as plt
from keras.applications import vgg19
from keras import backend as K

show_style_loss = 0
show_content_loss = 0
base_image_path ='base.jpg'
style_img = ['thunder.jpg','fire.jpg','water.jpg']

result_prefix = 'testImg_'



iterations = 1 
total_variation_weight = 1.0 
style_weight =1.0 
content_weight = 0.0001 
width, height = load_img(base_image_path).size
img_nrows = 400
img_ncols =  400

# util function to open, resize and format pictures into appropriate tensors

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

# util function to convert a tensor into a valid image


def deprocess_image(x):
    x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

base_image = K.variable(preprocess_image(base_image_path))
combination_image = K.placeholder((1, img_nrows, img_ncols, 3))
input_tensor = K.concatenate([base_image,combination_image],axis=0)

# concatenate style img
for img in style_img:
    img = K.variable(preprocess_image(img))
    input_tensor = K.concatenate([input_tensor,img],axis=0)

model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)

outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

def gram_matrix(x):
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))



def content_loss(base, combination):
    return K.sum(K.square(combination - base))

def content_compare(base1,base2):
    return K.sum(K.square(base1 - base2))



def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


loss = K.variable(0.)
# compare style difference
style_l = K.variable(0.)
# compare content difference
content_l = K.variable(0.)


layer_features = outputs_dict['block4_conv2']
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[1, :, :, :]
loss += content_weight * content_loss(base_image_features,
                                      combination_features)

feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']

def get_style_loss(loss,layer_features):
    combination_features = layer_features[1,:,:,:]
    for i in range(len(style_img)):
        sl = style_loss(layer_features[i+2,:,:,:],combination_features)
        loss += (style_weight / len(feature_layers)) * sl
    return loss

def compare_style_loss(loss,layer_features):
    sl = style_loss(layer_features[2,:,:,:],layer_features[0,:,:,:])
    return sl 

if show_content_loss == 1:
    content_l = content_weight * content_compare(layer_features[0,:,:,:],layer_features[2,:,:,:])

for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    loss = get_style_loss(loss,layer_features)
    # compare style loss
    if show_style_loss == 1:
        style_l = compare_style_loss(style_l,layer_features)


loss = loss / len(style_img)
loss += total_variation_weight * total_variation_loss(combination_image)


content_loss_output = [content_l]
style_loss_output = [style_l]

# get the gradients of the generated image wrt the loss
grads = K.gradients(loss, combination_image)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)
f_style_loss_outputs = K.function([combination_image],style_loss_output)
f_content_loss_outputs = K.function([combination_image],content_loss_output)

def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    if show_content_loss == 1:
        print("The content loss between base image and original image",f_content_loss_outputs([x]))
    if show_style_loss == 1:
        print("The style loss between base image and style image",f_style_loss_outputs([x]))
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

x = preprocess_image(base_image_path)

maxfun_num = 20
if show_content_loss == 1 or show_style_loss == 1:
    maxfun_num = 0

for i in range(iterations):
    print('Start of iteration', i)
    start = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=maxfun_num)
    end = time.time()
    print('time = ' , float(end - start))
    # save current generated image
    img = deprocess_image(x.copy())
    fname = result_prefix + '_at_iteration_%d.png' % i
    imsave(fname, img)

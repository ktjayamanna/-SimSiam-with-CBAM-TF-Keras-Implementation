# -*- coding: utf-8 -*-
"""
@author: kjayamanna
@Description: This is the code for the CBAM
@Reference: https://github.com/kobiso/CBAM-keras
"""
#%%
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Concatenate, multiply, Reshape
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Concatenate, multiply, Reshape
from tensorflow.keras.layers import Add, Lambda, GlobalMaxPooling2D, Activation
from keras import backend as K
from tensorflow.keras.models import Sequential, Model, load_model
plt.close('all')

# %%
def channel_attention(input_feature, ratio=1):
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature.shape[channel_axis]
	
	shared_layer_one = Dense(channel//ratio,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	
	avg_pool = GlobalAveragePooling2D()(input_feature)    
	avg_pool = Reshape((1,1,channel))(avg_pool)
	# assert avg_pool.shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	# assert avg_pool.shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	# assert avg_pool.shape[1:] == (1,1,channel)
	
	max_pool = GlobalMaxPooling2D()(input_feature)
	max_pool = Reshape((1,1,channel))(max_pool)
	# assert max_pool.shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	# assert max_pool.shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	# assert max_pool.shape[1:] == (1,1,channel)
	
	cbam_feature = Add()([avg_pool,max_pool])
	cbam_feature = Activation('sigmoid')(cbam_feature)
	
	# if K.image_data_format() == "channels_first":
	#       cbam_feature = Permute((3, 1, 2))(cbam_feature)
	
	return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
	kernel_size = 7
	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(input_feature)
	# assert avg_pool.shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(input_feature)
	# assert max_pool.shape[-1] == 1
	
	
	concat = Concatenate(axis=3)([avg_pool, max_pool])
	# assert concat.shape[-1] == 2
	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='sigmoid',
					kernel_initializer='he_normal',
					use_bias=False)(concat)	
	# assert cbam_feature.shape[-1] == 1	
	return multiply([input_feature, cbam_feature])

def cbam_block(cbam_feature, ratio=1):
	 cbam_feature = channel_attention(cbam_feature, ratio)
	 cbam_feature = spatial_attention(cbam_feature)
	 return cbam_feature


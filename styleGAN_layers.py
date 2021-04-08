#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, initializers
from tensorflow.keras.initializers import VarianceScaling


# In[2]:



# def _compute_fans(shape):
#   """Computes the number of input and output units for a weight shape.

#   Args:
#     shape: Integer shape tuple or TF tensor shape.

#   Returns:
#     A tuple of integer scalars (fan_in, fan_out).
#   """
#   if len(shape) < 1:  # Just to avoid errors for constants.
#     fan_in = fan_out = 1
#   elif len(shape) == 1:
#     fan_in = fan_out = shape[0]
#   elif len(shape) == 2:
#     fan_in = shape[0]
#     fan_out = shape[1]
#   else:
#     # Assuming convolution kernels (2D, 3D, or more).
#     # kernel shape: (..., input_depth, depth)
#     receptive_field_size = 1
#     for dim in shape[:-2]:
#       receptive_field_size *= dim
#     fan_in = shape[-2] * receptive_field_size
#     fan_out = shape[-1] * receptive_field_size
#   return int(fan_in), int(fan_out)


# In[8]:


class Conv2D(layers.Conv2D):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 use_bias=True,
                 activation=tf.nn.leaky_relu,
                 gain=1.0,
                 lrmul=1.0,
                 **kargs):
        super().__init__(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,
                         use_bias=use_bias,activation=activation,**kargs)
        
        self.lrmul=lrmul
        self.gain=gain
        
    def build(self,input_shape):
        input_shape=tf.TensorShape(input_shape)
        fan_in=tf.reduce_prod(self.kernel_size) * self._get_input_channel(input_shape) #计算fan_in,dtype=int32
        fan_in=tf.cast(fan_in,tf.float32)
        he_std=self.gain / tf.sqrt(fan_in)
        
        # Equalized learning rate and custom learning rate multiplier.
        #init_std=1.0 / self.lrmul
        self.runtime_coef=self.runtime_coef=tf.Variable(he_std * self.lrmul,name=self.name+'_coef',trainable=True)
        
        self.kernel_initializer=tf.keras.initializers.RandomNormal(0.0,1.0 / self.lrmul)
        super().build(input_shape)
        
    def call(self,inputs):
        x=self._convolution_op(inputs,self.kernel * self.runtime_coef)
        if self.use_bias:
            x=tf.nn.bias_add(x,self.bias * self.lrmul,data_format='NHWC')
        if self.activation is not None:
            x=self.activation(x)
        return x


# In[5]:


class Dense(layers.Dense):
    def __init__(self,
                 filters,
                 use_bias=True,
                 activation=tf.nn.leaky_relu,
                 gain=1.0,
                 lrmul=1.0,
                 **kargs):
        super().__init__(filters,use_bias=use_bias,activation=activation,**kargs)
        
        self.lrmul=lrmul
        self.gain=gain
        
    def build(self,input_shape):
        fan_in=input_shape[-1] #计算fan_in,dtype=int32
        fan_in=tf.cast(fan_in,tf.float32)
        he_std=self.gain / tf.sqrt(fan_in)
        
        # Equalized learning rate and custom learning rate multiplier.
        #init_std=1.0 / self.lrmul
        self.runtime_coef=tf.Variable(he_std * self.lrmul,name=self.name+'_coef',trainable=True)
        
        #Lecun初始化调用VarianceScaling
        #self.kernel_initializer=VarianceScaling(scale=1.0,mode='fan_in',distribution='truncated_normal')
        
        self.kernel_initializer=tf.keras.initializers.RandomNormal(0.0,1.0 / self.lrmul)
        
        super().build(input_shape)
        
    def call(self,inputs):
        rank=inputs.shape.rank
        if rank == 2:
            x=tf.matmul(inputs,self.kernel * self.runtime_coef)
        else:
            x=tf.tensordot(inputs,self.kernel * self.runtime_coef, axes=[[rank-1],[0]])
        
        if self.bias is not None:
            x=tf.nn.bias_add(x,self.bias * self.lrmul,data_format='NHWC')
        if self.activation is not None:
            x=self.activation(x)
        return x  


# In[7]:


# model=tf.keras.Sequential([
#     layers.Input([4,4,3]),
#     layers.Conv2D(16,kernel_size=-1),
# ])
# model.summary()


# In[ ]:





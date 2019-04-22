{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "#Import python package \n",
    "from keras import datasets \n",
    "import matplotlib.pyplot as plt\n",
    "import numpy\n",
    "from keras.utils import to_categorical\n",
    "from keras import models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(60000, 28, 28)\n",
      "(60000,)\n",
      "(10000, 28, 28)\n",
      "(10000,)\n"
     ]
    }
   ],
   "source": [
    "#load the data\n",
    "(trainimg, trainlb),(testimg,testlb) = datasets.mnist.load_data()\n",
    "print(trainimg.shape)\n",
    "print(trainlb.shape)\n",
    "print (testimg.shape)\n",
    "print(testlb.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(trainimg[10].reshape(28,28),cmap='gray')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3\n"
     ]
    }
   ],
   "source": [
    "print(trainlb[10])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "trainimg = trainimg.reshape(60000,28,28,1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Scale image\n",
    "\n",
    "trainimg = trainimg/255"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(60000, 10)\n"
     ]
    }
   ],
   "source": [
    "#onehot enchoding of labels\n",
    "trainlb = to_categorical(trainlb)\n",
    "print(trainlb.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Convolution Neural Network\n",
    "model = models.Sequential()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras import layers\n",
    "#add the first layer\n",
    "model.add(layers.Conv2D(filters=10,kernel_size=(3,3),\n",
    "                       input_shape=(28,28,1), activation =\"relu\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "#maxpooling\n",
    "model.add(layers.MaxPooling2D(pool_size=(2,2)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(layers.Conv2D(filters=10,kernel_size=(3,3),\n",
    "                     activation =\"relu\"))\n",
    "\n",
    "#maxpooling\n",
    "model.add(layers.MaxPooling2D(pool_size=(2,2)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(layers.Flatten())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "#add the hidden layer\n",
    "model.add(layers.Dense(100,activation='relu'))\n",
    "\n",
    "#add the output layer\n",
    "\n",
    "model.add(layers.Dense(10,activation='softmax'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Compile the neural network \n",
    "model.compile(loss=\"categorical_crossentropy\",metrics=['accuracy'],optimizer='adam')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Users\\VAIO\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\ops\\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Use tf.cast instead.\n",
      "Train on 48000 samples, validate on 12000 samples\n",
      "Epoch 1/5\n",
      "48000/48000 [==============================] - 21s 434us/step - loss: 1.3712 - acc: 0.6638 - val_loss: 0.4658 - val_acc: 0.8665\n",
      "Epoch 2/5\n",
      "48000/48000 [==============================] - 18s 382us/step - loss: 0.3762 - acc: 0.8906 - val_loss: 0.2752 - val_acc: 0.9212\n",
      "Epoch 3/5\n",
      "48000/48000 [==============================] - 18s 383us/step - loss: 0.2601 - acc: 0.9240 - val_loss: 0.2139 - val_acc: 0.9388\n",
      "Epoch 4/5\n",
      "48000/48000 [==============================] - 18s 380us/step - loss: 0.2061 - acc: 0.9385 - val_loss: 0.1793 - val_acc: 0.9500\n",
      "Epoch 5/5\n",
      "48000/48000 [==============================] - 19s 388us/step - loss: 0.1728 - acc: 0.9481 - val_loss: 0.1539 - val_acc: 0.9561\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x13f54f98>"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#train the model\n",
    "model.fit(trainimg,trainlb,epochs=5,batch_size=1000,validation_split=0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "What I Know\n",
      "For Sure\n",
      "\n",
      "*\n",
      "\n",
      "Oprah Winfrey\n"
     ]
    }
   ],
   "source": [
    "#import python package\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy\n",
    "import pytesseract\n",
    "import cv2\n",
    "import nltk\n",
    "import textblob\n",
    "\n",
    "#load a file and transver to gray\n",
    "img = cv2.imread(r\"C:\\Users\\VAIO\\Desktop\\AI\\AI Project\\what.jpg\")\n",
    "img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)\n",
    "img = cv2.threshold(img,0,255, cv2.THRESH_OTSU)[1]\n",
    "img = cv2.erode(img,kernel = numpy.ones(6))\n",
    "img = cv2.dilate(img,kernel = numpy.ones(6))\n",
    "plt.imshow(img,cmap='gray')\n",
    "plt.show()\n",
    "\n",
    "\n",
    "text = pytesseract.image_to_string(img)\n",
    "print(text)\n",
    "\n",
    "#Identify & get the text\n",
    "def get_right_word(text):\n",
    "    words = nltk.word_tokenize(text)\n",
    "    text2 = []\n",
    "    for word in words:\n",
    "        w = textblob.Word(word)\n",
    "        if len(w.definitions)>0:\n",
    "            text2.append(word)\n",
    "    return \" \".join(text2)\n",
    " \n",
    "get_right_word(text)\n",
    "\n",
    "# text to audio\n",
    "import pyttsx3\n",
    "engine = pyttsx3.init()\n",
    "engine.say(text)\n",
    "engine.runAndWait()"
   ]
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

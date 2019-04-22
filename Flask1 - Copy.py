{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app \"__main__\" (lazy loading)\n",
      " * Environment: production\n",
      "   WARNING: Do not use the development server in a production environment.\n",
      "   Use a production WSGI server instead.\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " * Restarting with stat\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "1",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[1;31mSystemExit\u001b[0m\u001b[1;31m:\u001b[0m 1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\VAIO\\Anaconda3\\lib\\site-packages\\IPython\\core\\interactiveshell.py:2969: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.\n",
      "  warn(\"To exit: use 'exit', 'quit', or Ctrl-D.\", stacklevel=1)\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, render_template,Response\n",
    "import cv2\n",
    "\n",
    "fd = cv2.CascadeClassifier(r\"haarcascadefiles/haarcascade_frontalface_alt.xml\")\n",
    "\n",
    "app=Flask(__name__)\n",
    "\n",
    "@app.route('/')\n",
    "def index():\n",
    "    return render_template(\"index.html\")\n",
    "\n",
    "def gen():\n",
    "    vid = cv2.VideoCapture(0)\n",
    "    while True:\n",
    "        ret,img = vid.read()\n",
    "        if ret ==True:\n",
    "            faces = fd.detectMultiScale(img,1.1,5)\n",
    "            if len(faces)>0:\n",
    "                for (x,y,w,h) in faces:\n",
    "                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)\n",
    "            ret,jpeg = cv2.imencode(\".jpg\",img)\n",
    "            yield (b'--frame\\r\\n'\n",
    "                  b'Content-Type: image/jpeg\\r\\n\\r\\n' + jpeg.tostring() + b'\\r\\n\\r\\n')\n",
    "        \n",
    "@app.route('/video_feed')\n",
    "def video_feed():\n",
    "    return Response(gen(),mimetype='multipart/x-mixed-replace;boundary=frame')\n",
    "\n",
    "if __name__==\"__main__\":\n",
    "    app.run(debug=True)\n",
    "    \n",
    "def capture_frame(self):\n",
    "        ret, frame = self.cap.read()\n",
    "        if ret:\n",
    "            _, frame = self.scanner.detect_edge(frame, True)\n",
    "            ret, jpeg = cv2.imencode('.jpg', frame)\n",
    "            self.transformed_frame = jpeg.tobytes()\n",
    "        else:\n",
    "            return None"
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

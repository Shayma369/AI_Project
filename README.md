{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Copy of README.md",
      "version": "0.3.2",
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "metadata": {
        "id": "uhgG2UGaTL5P",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "YvTE1WeeTROa",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "**Book Scanner**\n",
        "\n",
        "\n",
        "\n",
        "---\n",
        "\n",
        "\n",
        "\n",
        "---\n",
        "\n",
        "\n",
        "*Getting started*\n",
        "\n",
        "\n",
        "These instructions will get you a copy of the project up and running on your local machine for\n",
        "development and testing purposes.\n",
        "\n",
        "\n",
        "*Prerequisites*\n",
        "\n",
        "\n",
        "This application “Book Scanner” built using OpenCV + Python. In order to run this application, you have to have the following in your local machine:\n",
        "- You have to install Anaconda\n",
        "- Updated internet browser\n",
        "- Pip version 19\n",
        "\n",
        "*Installing*\n",
        "\n",
        "We need to Import all required packages &amp; libraries:\n",
        "- Opencv2\n",
        "- Jumpy\n",
        "- matplotlib.pyplot\n",
        "- matplotlib.image\n",
        "- pytesseract\n",
        "- nltk\n",
        "- textblob\n",
        "- fpdf\n",
        "\n",
        "\n",
        "*Application description*\n",
        "\n",
        "- You/ we need to import all required packages &amp; libraries\n",
        "- Write a code which set and open a webcam\n",
        "- Then a code to release a webcam\n",
        "- After that, a small code to show the image which has been captured\n",
        "- Then, write a code for image processing: First the picture will be processed from colored to gray “gray scale”, then black &amp; white.\n",
        "- After that take out all the noises from the captured picture.\n",
        "- Next, the code will read the text which has been scanned from the captured picture, and print it/ show it. (capture and print all correct words as per the dictionary)\n",
        "- Then, write a code for converting text to audio\n",
        "- Following, as we imported “fpdf” library, we can write a code to convert the captured\n",
        "picture to PDF. Also, you/ we can set the font type, font size, font style, PDF title and\n",
        "its alignments.\n",
        "- After that, we need to create a webpage to run the application on it, so we have to have a flask file which contains HTML codes.\n",
        "- In order to start work with Flask, go to Anaconda Prompt CMD, write:\n",
        "      - CD (put the file path)\n",
        "      - Pip install flask\n",
        "      - Python (flask’s file name)\n",
        "- In flask HTML codes, we can set the alignment of the camera box, capture button, scanned\n",
        "text.\n",
        "- At the end the PDF file will be saved in the same location of the project file.\n",
        "\n",
        "\n",
        "*Acknowledgment*\n",
        "\n",
        "\n",
        "Our sincere thanks to our instructor Mr. Anshu Pandey – who help us a lot on coding, and\n",
        "always making sure that we are understanding what we are working on."
      ]
    }
  ]
}
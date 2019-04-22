
**Book Scanner**



---



---


*Getting started*


These instructions will get you a copy of the project up and running on your local machine for
development and testing purposes.


*Prerequisites*


This application “Book Scanner” built using OpenCV + Python. In order to run this application, you have to have the following in your local machine:
- You have to install Anaconda
- Updated internet browser
- Pip version 19

*Installing*

We need to Import all required packages &amp; libraries:
- Opencv2
- Jumpy
- matplotlib.pyplot
- matplotlib.image
- pytesseract
- nltk
- textblob
- fpdf


*Application description*

- You/ we need to import all required packages &amp; libraries
- Write a code which set and open a webcam
- Then a code to release a webcam
- After that, a small code to show the image which has been captured
- Then, write a code for image processing: First the picture will be processed from colored to gray “gray scale”, then black &amp; white.
- After that take out all the noises from the captured picture.
- Next, the code will read the text which has been scanned from the captured picture, and print it/ show it. (capture and print all correct words as per the dictionary)
- Then, write a code for converting text to audio
- Following, as we imported “fpdf” library, we can write a code to convert the captured
picture to PDF. Also, you/ we can set the font type, font size, font style, PDF title and
its alignments.
- After that, we need to create a webpage to run the application on it, so we have to have a flask file which contains HTML codes.
- In order to start work with Flask, go to Anaconda Prompt CMD, write:
      - CD (put the file path)
      - Pip install flask
      - Python (flask’s file name)
- In flask HTML codes, we can set the alignment of the camera box, capture button, scanned
text.
- At the end the PDF file will be saved in the same location of the project file.


*Acknowledgment*


Our sincere thanks to our instructor Mr. Anshu Pandey – who help us a lot on coding, and
always making sure that we are understanding what we are working on.

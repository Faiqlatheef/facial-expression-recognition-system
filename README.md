# facial-expression-recognition-system
Facial Expression Recognition :
So let’s talk about the ways we could go about recognizing someone’s facial expressions. We will look at classical approaches first.
Haar Cascades based Recognition:
Perhaps the oldest method that could work are Haar Cascades. So essentially these Haar Cascades also called viola jones Classifier is an outdated Object detection technique by Paul Viola and Michael Jones in 2001. It is a machine learning-based approach where a cascade is trained from a lot of positive and negative images. It is then used to detect objects in images.
The most popular use of these cascades is as a face detector which is still used today, although there are better methods available. 
Now instead of using face detection, we could train a cascade to detect expressions. Since you can only train a single class with a cascade so you’ll need multiple cascades. A better way to go about is to first perform face detection then look for different features inside the face ROI, like detecting a smile with this smile detection cascade. You can also train a frown detector and so on.
Truth be told, this method is so weak that I wouldn’t even try experimenting with this in this time and era but since people have used this in the past so I’m just putting it there.

if you want to use the sysytem first unzip the file and save in the same folder then train it and use it

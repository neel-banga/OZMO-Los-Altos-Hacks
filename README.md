
## MADE FOR LOS ALTOS HACKATHON - 1ST PLACE 

#### Devpost Submission: https://devpost.com/software/ozmo


##### What does it do?
Our inspiration is to harness the power of technology to make a positive impact on the lives of visually impaired individuals. By utilizing AI, machine learning, computer vision, and other relevant technologies, our goal is to create a user-friendly solution that promotes inclusivity, equality, and independence for blind individuals. We are committed to leveraging technology to enhance accessibility and empower visually impaired individuals to navigate their environment, access information, and perform daily tasks with confidence and independence.


##### Features

*OBJECT DETECTION - Using the YOLO model, we were able to use the powers of PyTorch, YOLO, and Python to detect objects in real-time. We utilized the positioning of the bounding boxes relative to the the center of the screen to understand the positioning of certain objects.
*TEXT MESSAGING - We utilized Twilio to allow users to send text messages to contacts using just their voice.  The API needs acess to our authorization ID and Token, this allows for personalization per user. When a new user signs up, we create a Twilio subprocess just for them.
*CALLING - Voice calling was quite tricky in comparison to text messaging, due to Twilio's lack of support for lifetime calling, we had to create a "dummy URL", then we curled the URL exposing it to the internet. This allowed both sides to communicate via HTTP requests. 
*ENHANCED READING - We used the powersd of PyTesseract and transfer learning to be able to read text in real-time.  To get our model running in real-time, we took advantage Open CV live-time video feed, we then pass frames of this feed to the neural network, which is able to convert an image to text.
*FACE RECOGNTION - We built a PyTorch neural network from scratch to be able to identify close friends and family that the user uploads to their account. The model retrains based on these user photos, currently, the model takes a while to train, so we plan to allow users to upload photos and then send them a ping through Twilio once the model has been adjusted.

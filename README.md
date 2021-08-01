# MIT FutureMakers

The MIT FutureMakers Create-a-thon is a virtual, part-time 6-week AI learning program, developed through a collaboration between [SureStart](https://mysurestart.com/) and the [MIT RAISE (Responsible AI for Social Empowerment and Education) Initiative](https://raise.mit.edu/).

The FutureMakers program, which started on July 6th, 2021 includes:
- Learning AI, machine learning and emotion AI concepts
- Turning skills into confidence through building AI solutions hands-on
- Consistent support and encouragement from a mentor 1-1
- Regular tech talks and career readiness seminars.
- Entrepreneurship and leadership skills development

This 6-week virtual program provided me with a unique opportunity to build AI solutions that tackle some of today’s most pressing challenges.

[Here is a list of all projects that I worked on during this program](#all-projects)

### July 2021
|           Mon          |           Tue          |           Wed          |           Thu          |           Fri          |Sat |Sun |
|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:--:|:--:|
|                        |                        |                        |           01           |           02           | 03 | 04 |
|           05           | [06](#day-01-06072021) | [07](#day-02-07072021) | [08](#day-03-08072021) | [09](#day-04-09072021) | 10 | 11 |
| [12](#day-07-12072021) | [13](#day-08-13072021) | [14](#day-09-14072021) | [15](#day-10-15072021) | [16](#day-11-16072021) | 17 | 18 |
| [19](#day-14-19072021) | [20](#day-15-20072021) | [21](#day-16-21072021) | [22](#day-17-22072021) | [23](#day-18-23072021) | 24 | 25 |
| [26](#day-21-26072021) | [27](#day-22-27072021) | [28](#day-23-28072021) |           29           |           30           | 31 |    |

## Day 01 (06.07.2021)
### ***What do you hope to learn from this program?***
I am looking forward to learning more about AI and its ethical impact on society. I am also hoping the apply the knowledge and skills learned during this program to hands-on project to ensure that I properly understand the material.

## Day 02 (07.07.2021)
### ***What did you learn during Dr. Kong's seminar?***
Dr. Kong's seminar on leadership taught me how to use storytelling to take action in my community and make a positive difference. 

There are 3 types of callings: A call of us (community), self (leadership), and now (strategy and action). Once you have decided on the calling, you craft a story that has three parts: a choice, a challenge, and an outcome. A moral at the end is a plus. 

This well-planned seminar inspired me to share my own story and experience to become a supportive and encouraging visionary leader. 

## Day 03 (08.07.2021)
### ***Lesson Plan***
- Reviewed ML models with this [article](https://www.toptal.com/machine-learning/machine-learning-theory-an-introductory-primer)
### ***Action Item:*** 
- [Decision Tree Classifier](/decision-tree-classifier/)

### ***What is the difference between supervised and unsupervised learning?***
| Supervised Learning                                                                      | Unsupervised Learning                                                                                    |
|------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| ***Use Cases:***<br>Predict outcomes for new data                                        | ***Use Cases:***<br>Gain insights from large volumes of data                                             |
| ***Applications:***<br>- Spam Detection<br>- Sentiment Analysis<br>- Pricing Predictions | ***Applications:***<br>- Anomaly Detection<br>- Recommendation Engines<br>- Medical Image Classification |
| ***Drawbacks:***<br>Can be time-consuming to label dataset and train models              | ***Drawbacks:***<br>Can have inaccurate results and reflects biases that might be present in the dataset |

### ***Describe why the following statement is FALSE: Scikit-Learn has the power to visualize data without a Graphviz, Pandas, or other data analysis libraries.***
The Scikit-Learn library is built on top of visualization libraries like Pandas and Graphviz. Therefore, data analysis libraries need to be installed prior to using Scikit-Learn.

### ***Supplemental Activity:*** 
- [Predicting Housing Prices](/predicting-housing-prices/)

## Day 04 (09.07.2021)
### ***Lesson Plan***
- Gained a high level understanding of DL models and algorithms through [this article](https://serokell.io/blog/deep-learning-and-neural-network-guide)
### ***Real World Problem:*** 
According to the [WHO](https://www.who.int/news-room/fact-sheets/detail/falls), an estimated 684 000 fatal falls occur each year, making it the second leading cause of unintentional injury death, after road traffic injuries. However, not all falls are fatal, with 37.3 million falls being severe enough to require medical attention. With so many people being injured or killed each year by falls, it is of great social significance to provide them with accurate, dependable, and effective procedures to mitigate the effects of falls. 

### ***Dataset:*** 
I am using the [SisFall dataset](http://sistemic.udea.edu.co/en/research/projects/english-falls/) which consists of data collected from two accelerometers and one gyroscope. This dataset is the only one I could find that includes falls by people over 60. However, due to medical concerns, there is a bias towards younger groups of people.

### ***Method:*** 
I'm currently developing a model to detect falls using deep learning, which will be deployed to an Arduino. The binary classification model uses a combination of a Convolutional Neural Network (CNN) and Long Short Term Memory (LSTM) for time series prediction. 

The biggest hurdle I am facing is the low processing power of the Arduino. I managed to train a model that can detect falls with over 99.5% accuracy, though the computation requirements for feature extraction proved to be too high for the Arduino to handle. I have therefore switched to using deep learning and am training the model on raw data. More information on the project can be found here: [https://github.com/mferuscomelo/fall-detection](https://github.com/mferuscomelo/fall-detection)

### ***Supplemental Activity:*** 
- Read this [article](https://blogs.nvidia.com/blog/2016/07/29/whats-difference-artificial-intelligence-machine-learning-deep-learning-ai/) on the difference between AI and ML.

## Day 07 (12.07.2021)
### ***Lesson Plan:*** 
- [TensorFlow Basics](/tensorflow-basics/)
- [Belgian Traffic Sign Classification](https://github.com/mferuscomelo/traffic-signs-classification)
- [Wine Identification](https://github.com/mferuscomelo/wine-identification)

### ***What are "Tensors" and what are they used for in Machine Learning?***
Tensors are data-structures that can be visualized as n-dimensional arrays, with n > 2. We only call structures with 3 dimensions or more "Tensors" so as to not confuse them with lower-dimensional structures such as matrices, vectors, and scalars.  

![Difference between a scalar, a vector, a matrix and a tensor](/images/scalar-vector-matrix-tensor.png "Difference between a scalar, a vector, a matrix and a tensor")

Tensors usually contain numerical data and are the backbone of neural networks. All transformations of a neural network can be reduced to tensor operations.

### ***What did you notice about the computations that you ran in the TensorFlow programs (i.e. interactive models) in the tutorial?***
The datasets had to be processed before training the model so that it could better identify the relationships between the data. This process is called feature extraction or feature engineering.

## Day 08 (13.07.2021)
### ***Lesson Plan:*** 
- Reviewed [this guide](https://serokell.io/blog/deep-learning-and-neural-network-guide) about common components of neural networks and how they work with different ML functions and algorithms. 
### ***Action Item:*** 
- [Sarcasm detection](https://github.com/mferuscomelo/sarcasm-detection)

## Day 09 (14.07.2021)
### ***Lesson Plan:*** 
- Learned about CNNs using this [cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)
- [Visualized](https://www.cs.ryerson.ca/~aharley/vis/conv/flat.html) how CNNs work with handwritten digits
- [What is a confusion matrix?](https://machinelearningmastery.com/confusion-matrix-machine-learning/)

### ***Action Item:*** 
- [MNIST Digits Classification](https://github.com/mferuscomelo/mnist-digits-classification)

## Day 10 (15.07.2021)
### ***Lesson Plan:*** 
- Reviewed [presentation](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture19-bias.pdf) on algorithmic bias

### ***Action Item:***
- Played [survival of the best fit](https://www.survivalofthebestfit.com/) to learn more about how AI might impact human resources and hiring processes in different fields

### ***How do you think Machine Learning or AI concepts were utilized in the design of this game?***
This game demonstrated the process of automating hiring decisions based on data which might be biased and its consequences. However, this topic isn't as far-fetched as some may think. In 2014, Amazon decided to try to automate hiring at their company<sup>[1](#footnote-1)</sup>. Just like in the game, the hiring process done by humans was already biased with the overwhelming majority of hired employees being male. Although the team developing the algorithm might not have intended for this bias to be present, the sheer number of resumes present in the training set created an algorithmic bias towards hiring male applicants. In real life, men get a lot more support for getting into STEM related fields, whereas women are often actively discouraged from those careers and instead, are taught to start jobs "meant for women."

In the same way, there was an abundance of orange candidates while hiring, showing that blue applicants were removed from the competition even before they could enter. Towards the end, it got so bad that the applicant pool of around 10 people was comprised entirely of orange people. Seeing the statistics of my hiring process in the end, around 75% of people hired **and** rejected were orange. 

This, combined with "Google's" dataset that wasn't inspected for bias, meant that the algorithm was hiring orange people with almost two times as often as blue people. 

### ***Can you give a real-world example of a biased machine learning model, and share your ideas on how you make this model more fair, inclusive, and equitable? Please reflect on why you selected this specific biased model.***
In 2015, Google's image classification algorithm for Google Photos misclassified a black couple as being gorillas<sup>[2](#footnote-2)</sup>. This racist classification was a result of algorithmic bias present in Google's ML model due to insufficient training data from a diverse group of people. 

This example was an event that showed me that algorithms aren't as infallible as we might think. They propagate biases present in the data they see, some of which might not be noticed by humans. In the informational game [survival of the best fit](https://www.survivalofthebestfit.com/), there was a slight bias towards hiring orange people and rejecting blue people. This, combined with the fact that "Google's" hiring processes was used in the dataset without checking for bias, meant that the algorithm trained on the data amplified these biases to the point of making orange people almost twice as likely as blue people to be hired.  

## Day 11 (16.07.2021)
### ***Lesson Plan:*** 
- Reviewed [CNN Architecture](https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939)

### ***Action Item:*** 
- Improved [MNIST Digit Classification](https://github.com/mferuscomelo/mnist-digits-classification) algorithm and added option to make predictions in the notebook

## Day 14 (19.07.2021)
### ***Lesson Plan:*** 
- Read [article](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/) on choosing loss functions
- Watched [lecture](https://www.youtube.com/watch?v=h7iBpEHGVNc) and reviewed [slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture3.pdf) on loss functions and optimization

### ***Action Item:*** 
- [Housing Prices Prediction](https://github.com/mferuscomelo/housing-price-prediction)

## Day 15 (20.07.2021)
### ***Lesson Plan:*** 
- Reviewed [article](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/) on choosing activation functions
- Learned [how to implement the ReLU activation function](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)

### ***Choosing an activation function for a hidden layer***
![Choosing an activation function for a hidden layer](/images/activation-function-hidden.png "Choosing an activation function for a hidden layer")

### ***Choosing an activation function for an output layer***
![Choosing an activation function for an output layer](/images/activation-function-output.png "Choosing an activation function for an output layer")

### ***What are some advantages of the Rectified Linear activation function? State a use case.***
The ReLU has become the most used activation function for hidden layers.The function is simple to use and efficient at overcoming the drawbacks of earlier popular activation functions such as sigmoid and tanh. It is less prone to vanishing gradients, which prohibit deep models from being trained, however it can suffer from other issues such as saturated or "dead" units.

The ReLU activation function can be used in hidden layers for multilayer perceptrons and convolutional neural networks.

## Day 16 (21.07.2021)
### ***Lesson Plan:*** 
- Reviewed [article](https://hub.packtpub.com/machine-learning-ethics-what-you-need-to-know-and-what-you-can-do/) on the importance of ethics in the real-world context of AI and automation.

### ***Action Item:*** 
- [Gender recognition from faces](https://github.com/mferuscomelo/gender-recognition-face)

## Day 17 (22.07.2021)
### ***Lesson Plan:*** 
- Reviewed different image classification techniques through this [article](https://iq.opengenus.org/basics-of-machine-learning-image-classification-techniques/)

### ***Action Item:*** 
- [Animal Classification](https://github.com/mferuscomelo/animal-classification)

## Day 18 (23.07.2021)
### ***Lesson Plan:*** 
- Learned [how to avoid overfitting](https://machinelearningmastery.com/introduction-to-regularization-to-reduce-overfitting-and-improve-generalization-error/)
- Read about the [ethics of machine learning](https://towardsdatascience.com/ethics-in-machine-learning-9fa5b1aadc12)

### ***Action Item:*** 
- [Sentiment Analysis](https://github.com/mferuscomelo/sentiment-analysis)

## Day 21 (26.07.2021)
### ***Lesson Plan:*** 
- Reviewed a [tutorial](https://machinelearningmastery.com/upsampling-and-transpose-convolution-layers-for-generative-adversarial-networks/) on upsampling
- Learned about [autoencoders](https://blog.keras.io/building-autoencoders-in-keras.html)

### ***Action Item:*** 
- [Autoencoders](https://github.com/mferuscomelo/autoencoders)

## Day 22 (27.07.2021)
### ***Lesson Plan:*** 
- Watched a [TED Talk](https://www.youtube.com/watch?v=ujxriwApPP4) on the origins of Affective Computing
- Read about the [EMPath Makeathon](https://mysurestart.com/case-study)

### ***Action Item:*** 
- [Speech Emotion Analyzer](https://github.com/mferuscomelo/speech-emotion-analyzer)

## Day 23 (28.07.2021)
### ***Lesson Plan:*** 
- Reviewed [this guide](https://medium.com/@calebkaiser/a-list-of-beginner-friendly-nlp-projects-using-pre-trained-models-dc4768b4bec0) on applied NLP projects

### ***Action Item:*** 
- [Movie Review Classifier](https://github.com/mferuscomelo/movie-review-classifier)

## All Projects
- [Flower Classification](/decision-tree-classifier/)
- [Predicting Housing Prices](/predicting-housing-prices/)
- [TensorFlow Basics](/tensorflow-basics/)
- [Belgian Traffic Sign Classification](https://github.com/mferuscomelo/traffic-signs-classification)
- [Wine Identification](https://github.com/mferuscomelo/wine-identification)
- [Sarcasm Detection](https://github.com/mferuscomelo/sarcasm-detection)
- [MNIST Digits Classification](https://github.com/mferuscomelo/mnist-digits-classification)
- [Housing Prices Classification](https://github.com/mferuscomelo/housing-price-prediction)
- [Gender Recognition from Faces](https://github.com/mferuscomelo/gender-recognition-face)
- [Animal Classification](https://github.com/mferuscomelo/animal-classification)
- [Sentiment Analysis](https://github.com/mferuscomelo/sentiment-analysis)
- [Autoencoders](https://github.com/mferuscomelo/autoencoders)
- [Speech Emotion Analyzer](https://github.com/mferuscomelo/speech-emotion-analyzer)
- [Movie Review Classifier](https://github.com/mferuscomelo/movie-review-classifier)

## Resources
### Cheatsheets
- [Loss functions cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)
- [CNN cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)
- [Activation functions cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html)
- [Pandas Cheatsheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [NLP Resources](https://towardsdatascience.com/how-to-get-started-in-nlp-6a62aa4eaeff)

### Datasets
- [Public datasets](https://docs.google.com/spreadsheets/d/1qYjOWt39m6r3DpMYx3kespQRVgT6icn5tALXHrztbY4/edit?usp=sharing)
- [Kaggle](https://www.kaggle.com/datasets)

## Sources
[Difference between a scalar, a vector, a matrix and a tensor](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.1-Scalars-Vectors-Matrices-and-Tensors/)

[Choosing an activation function for a hidden layer](https://www.kaggle.com/discussion/212325)

[Choosing an activation function for an output layer](https://www.kaggle.com/discussion/212325)

<a id="footnote-1">1</a>: Goodman, Rachel. “Why Amazon's Automated Hiring Tool Discriminated Against Women.” American Civil Liberties Union, American Civil Liberties Union, 15 Oct. 2018, www.aclu.org/blog/womens-rights/womens-rights-workplace/why-amazons-automated-hiring-tool-discriminated-against.  

<a id="footnote-2">2</a>: “Google Apologizes for Photos App's Racist Blunder.” BBC News, BBC, 1 July 2015, www.bbc.com/news/technology-33347866. 
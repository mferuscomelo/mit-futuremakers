# MIT FutureMakers

## Day 1 (06.07.2021)
I am looking forward to learning more about AI and its ethical impact on society. I am also hoping the apply the knowledge and skills learned during this program to hands-on project to ensure that I properly understand the material.

## Day 2 (07.07.2021)
Dr. Kong's seminar on leadership taught me how to use storytelling to take action in my community and make a positive difference. 

There are 3 types of callings: A call of us (community), self (leadership), and now (strategy and action). Once you have decided on the calling, you craft a story that has three parts: a choice, a challenge, and an outcome. A moral at the end is a plus. 

This well-planned seminar inspired me to share my own story and experience to become a supportive and encouraging visionary leader. 

## Day 3 (08.07.2021)
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

## Day 4 (09.07.2021)
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

### Day 7 (12.07.2021)
#### ***Lesson Plan:*** 
- [TensorFlow Basics](/tensorflow-basics/)
- [Belgian Traffic Sign Classification](https://github.com/mferuscomelo/traffic-signs-classification)
- [Wine Identification](https://github.com/mferuscomelo/wine-identification)

### ***What are "Tensors" and what are they used for in Machine Learning?***
Tensors are data-structures that can be visualized as n-dimensional arrays, with n > 2. We only call structures with 3 dimensions or more "Tensors" so as to not confuse them with lower-dimensional structures such as matrices, vectors, and scalars.  

![Difference between a scalar, a vector, a matrix and a tensor](/images/scalar-vector-matrix-tensor.png "Difference between a scalar, a vector, a matrix and a tensor")

Tensors usually contain numerical data and are the backbone of neural networks. All transformations of a neural network can be reduced to tensor operations.

### ***What did you notice about the computations that you ran in the TensorFlow programs (i.e. interactive models) in the tutorial?***
The datasets had to be processed before training the model so that it could better identify the relationships between the data. This process is called feature extraction or feature engineering.

### Day 8 (13.07.2021)
#### ***Lesson Plan:*** 
- Reviewed [this guide](https://serokell.io/blog/deep-learning-and-neural-network-guide) about common components of neural networks and how they work with different ML functions and algorithms. 
### ***Action Item:*** 
- [Sarcasm detection](https://github.com/mferuscomelo/sarcasm-detection)

### Day 9 (14.07.2021)
#### ***Lesson Plan:*** 
- Learned about CNNs using this [cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)
- [Visualized](https://www.cs.ryerson.ca/~aharley/vis/conv/flat.html) how CNNs work with handwritten digits
- [What is a confusion matrix?](https://machinelearningmastery.com/confusion-matrix-machine-learning/)

#### ***Action Item:*** 
- [MNIST Digits Classification](https://github.com/mferuscomelo/mnist-digits-classification)

## Sources
[Difference between a scalar, a vector, a matrix and a tensor](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.1-Scalars-Vectors-Matrices-and-Tensors/)
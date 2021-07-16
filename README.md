# MIT FutureMakers

### July 2021
|           Mon          |           Tue          |           Wed          |           Thu          |           Fri          |Sat |Sun |
|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:--:|:--:|
|                        |                        |                        |           01           |           02           | 03 | 04 |
|           05           | [06](#day-01-06072021) | [07](#day-02-07072021) | [08](#day-03-08072021) | [09](#day-04-09072021) | 10 | 11 |
| [12](#day-07-12072021) | [13](#day-08-13072021) | [14](#day-09-14072021) | [15](#day-10-15072021) |           16           | 17 | 18 |
|           19           |           20           |           21           |           21           |           23           | 24 | 25 |
|           26           |           27           |           28           |           29           |           30           | 31 |    |

## Day 01 (06.07.2021)
I am looking forward to learning more about AI and its ethical impact on society. I am also hoping the apply the knowledge and skills learned during this program to hands-on project to ensure that I properly understand the material.

## Day 02 (07.07.2021)
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

## Sources
[Difference between a scalar, a vector, a matrix and a tensor](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.1-Scalars-Vectors-Matrices-and-Tensors/)

<a id="footnote-1">1</a>: Goodman, Rachel. “Why Amazon's Automated Hiring Tool Discriminated Against Women.” American Civil Liberties Union, American Civil Liberties Union, 15 Oct. 2018, www.aclu.org/blog/womens-rights/womens-rights-workplace/why-amazons-automated-hiring-tool-discriminated-against.  

<a id="footnote-2">2</a>: “Google Apologizes for Photos App's Racist Blunder.” BBC News, BBC, 1 July 2015, www.bbc.com/news/technology-33347866. 
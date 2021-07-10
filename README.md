# MIT FutureMakers

## Responses

### Day 1 (06.07.2021)
I am looking forward to learning more about AI and its ethical impact on society. I am also hoping the apply the knowledge and skills learned during this program to hands-on project to ensure that I properly understand the material.

### Day 2 (07.07.2021)
Dr. Kong's seminar on leadership taught me how to use storytelling to take action in my community and make a positive difference. 

There are 3 types of callings: A call of us (community), self (leadership), and now (strategy and action). Once you have decided on the calling, you craft a story that has three parts: a choice, a challenge, and an outcome. A moral at the end is a plus. 

This well-planned seminar inspired me to share my own story and experience to become a supportive and encouraging visionary leader. 

### Day 3 (08.07.2021)
#### **What is the difference between supervised and unsupervised learning?**
| Supervised Learning                                                                    | Unsupervised Learning                                                                                  |
|----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| **Use Cases:**<br>Predict outcomes for new data                                        | **Use Cases:**<br>Gain insights from large volumes of data                                             |
| **Applications:**<br>- Spam Detection<br>- Sentiment Analysis<br>- Pricing Predictions | **Applications:**<br>- Anomaly Detection<br>- Recommendation Engines<br>- Medical Image Classification |
| **Drawbacks:**<br>Can be time-consuming to label dataset and train models              | **Drawbacks:**<br>Can have inaccurate results and reflects biases that might be present in the dataset |

#### **Describe why the following statement is FALSE: Scikit-Learn has the power to visualize data without a Graphviz, Pandas, or other data analysis libraries.**
The Scikit-Learn library is built on top of visualization libraries like Pandas and Graphviz. Therefore, data analysis libraries need to be installed prior to using Scikit-Learn.

### Day 4 (09.07.2021)
I'm currently developing a model to detect falls using deep learning, which will be deployed to an Arduino. The binary classification model uses a combination of a Convolutional Neural Network (CNN) and Long Short Term Memory (LSTM) for time series prediction. I am using the [SisFall dataset](http://sistemic.udea.edu.co/en/research/projects/english-falls/) which consists of data collected from two accelerometers and one gyroscope.

The biggest hurdle I am facing is the low processing power of the Arduino. I managed to train a model that can detect falls with over 99.5% accuracy, though the computation requirements for feature extraction proved to be too high for the Arduino to handle. I have therefore switched to using deep learning and am training the model on raw data. More information on the project can be found here: [https://github.com/mferuscomelo/fall-detection](https://github.com/mferuscomelo/fall-detection)
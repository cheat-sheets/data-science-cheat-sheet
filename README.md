# Data Science Cheet Sheet

## Deep Learning

### Structuring ML Projects

It is better to find a single optimization metric, this way it will be easier to choose a better model.
When it's not possible to choose a single optimization metric, you can add **satisfying** metrics. 
For example, error rate is an optimization metric and the time it takes to run the classification on an object is a 
satisfying metric.
 
**Bayes error rate** is the lowest possible error rate for any classifier of a random outcome 
(into, for example, one of two categories) and is analogous to the irreducible error.

If your algorithm is performing worse than a human, then to improve your algorithm you can:
- Get labeled data from humans.
- Gain insight from manual error analysis: why did a person get this right?
- Better analysis of bias/variance.

**When to focus on bias and when on variance**:
- If human error is **1%**, train error is **8%**, dev error is **10%**, then focus on **avoidable bias**, i.e reducing the train error,
    because it can potentially be reduced by 7 pp, compared to just 2 pp for dev error. To reduce the train error you can try the following:
        - Train a bigger model
        - Train longer
        - Try better optimization algorithms: RMSProp, Adam
        - Try a different NN architecture: RNN, CNN
        - Hyperparameter search 
- If human error is **7%**, train error is **8%**, dev error is **10%**, then focus on **variance**, i.e reducing the dev error,
    because it can potentially be reduced by 2 pp, compared to just 1 pp for train error. To reduce the dev error you can try the following:
        - Add more data
        - Regularization: L1, L2, dropout, 
        - Data augmentation
        - Try a different NN architecture: RNN, CNN
        - Hyperparameter search 

Problems where ML significantly surpasses human-level performance:
- Online advertising
- Product recommendations
- Logistics (predicting transit time)
- Loan approvals

All of the above are **ML on structured data** as opposed to natural perception.

Advice from Andrej Karpathy for learning ML: try implementing a NN from scratch, without relying on any libraries
like TensorFlow. This will help you learn how deep learning works under the hood.

**Error Analysis**: focus on what contributes most to the algorithm error. For example, if 90% of errors are due to
blurry images and 10% are due to misclassified as a dog instead of a cat, then focus on blurry images to reduce the error.  

**Incorrectly labeled data**: 
- fix it if it contributes a significant portion to error; 
- fix it across train, dev, test datasets universally. Otherwise it may introduce bias to the dataset.

It's important to make your dev and test datasets be as close to real-world data, even if it results in 
train and dev/test datasets be drawn from different distributions. This way you optimise to the right target. 
In this case to perform bias/variance analysis introduce train-dev dataset, to measure the variance contribution 
to error.

![data-mismatch.png](./assets/data-mismatch.png)

**Transfer Learning** - using intermediate NN layers, that were pre-trained on some problem A, for a different problem B.
For example problem A can be classifying cats and dogs, problem B can be classifying lung desease in radiology images.
It makes sense when:
- Problems A and B have the same input.
- There is a lot more input for problem A than for problem B.
- Low level features from A could be helpful for learning B.    

**Multi-task Learning** - training a NN for a classification problem where an input can be assigned multiple classes,
for example an image which can contain cars, pedestrians, stop signs, traffic lights, or any combination of those.
It can give better results than training a separate NN for each class, because the intermediate layers are reused.

**End-to-end ML** - solving a problem using just an ML algorithm without any hand-designed components as part of the 
whole system. For example, for a **speech recognition** task an end-to-end ML approach is to use audio as an input for
an ML algorithm and the transcript as the output, as opposed to manually extracting features from the audio first,
then phonemes, the words, and then generating a transcript.
- Pros: let the data speak, less hand-designing of components needed.
- Cons: may need large amount of data, excludes potentially useful hand-designed components.

## Random Notes

- Train, dev, and test datasets:
    - Dev dataset prevents overfitting NN parameters (weights and biases) to the train data
    - Test dataset prevents overfitting NN hyper-parameters (model architecture, number of layers types of layers) 
        to the train and dev data.
    
- What stage are we at? Stages of an ML project:
    - 1. Individual contributor
    - 2. Delegation
    - 3. Digitization
    - 4. Big Data and Analytics
    - 5. Machine Learning 

- CRISP-DM model
    - 1. Business understanding
    - 2. Data understanding
    - 3. Data preparation
    - 4. Modeling
    - 5. Evaluation
    - 6. Deployment  
    
- Precision, recall, accuracy, sensitivity, specificity
    - Precision and recall https://en.wikipedia.org/wiki/Precision_and_recall
    - Accuracy = Sensitivity * prevalence + Specificity * (1 - prevalence)
    - F1-score = 2 / ((1/P) + (1/R)) - harmonic mean, average speed.
    
![precisionrecall.png](./assets/precisionrecall.png)    

## Google Colab Notebooks

- [10 Minutes to Pandas](https://colab.research.google.com/drive/1LuUxoGo1yELpJ9JKnQF2Mujdm_9RYVqR)
- [Keras Hello World](https://colab.research.google.com/drive/14D_1LHcgFdTjmHzDtJV3M6h4flSR-pyK)
- [Fashion MNIST image classification with intermediate layers visualization](https://colab.research.google.com/drive/1ZPipu8FLPMf4sZ3E-v3rocmXC-58sRV_)
- Natality dataset in BigQuery:
    - [Exploring natality dataset](https://colab.research.google.com/drive/1LRNXqmjURFwMyjyrHgptsZ8vozYafefU)
    - [Creating a sampled dataset](https://colab.research.google.com/drive/1VQBd37-EVw9z5To4o8GE6oX8iHqcL7c-)
    - [DNN to predict baby weight with TensorFlow](https://colab.research.google.com/drive/1xbEL-0gaEmq-CyTfGuC_CAs5vpCmrWdi)
    - [DNN to predict baby weight with Keras](https://colab.research.google.com/drive/1ebQ7nTqx_f6VpkBPi0MMB8QFrQHy2U9M)

## Resources

- Deep Learning Specialization https://www.coursera.org/specializations/deep-learning
- Machine Learning with TensorFlow on GCP Specialization https://www.coursera.org/specializations/machine-learning-tensorflow-gcp
- Advanced Machine Learning with TensorFlow on GCP Specialization https://www.coursera.org/specializations/advanced-machine-learning-tensorflow-gcp
- TensorFlow and Keras in Practice Specialization https://www.coursera.org/specializations/tensorflow-in-practice
- Neural Networks and Deep Learning, book by Michael, Nielsen http://neuralnetworksanddeeplearning.com/


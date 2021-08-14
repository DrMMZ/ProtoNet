# ProtoNet for Few-Shot Learning

This repository is a TensorFlow2 implementation of [ProtoNet](https://arxiv.org/abs/1703.05175) (Prototypical Network) and its applications, aiming for creating a tool in zero/few-shot learning task that can be easily extended to other datasets or used in building projects. It includes

1. source code of ProtoNet and its configuration (multiple GPUs training, inference and evaluation);
2. source code of data (ProtoNet's inputs) generator using multiple CPU cores; 
3. source code of two backbones: conv4 (original in paper) and resnet;
4. source code of utilities such as image preprocessing and dataset.

### Applications

1. By just learning few face images from a random person, the model is able to identify and recognize that person effectively from a group of people. Below are samples tested on the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.
<p align="center">
  <img src="https://raw.githubusercontent.com/DrMMZ/drmmz.github.io/master/images/celeba/7.JPG" width='380' height='420'/>
  <img src="https://raw.githubusercontent.com/DrMMZ/drmmz.github.io/master/images/celeba/12.JPG" width='380' height='420'/>
</p> 
In each sample, there are 3 face images learned by the model (under the text "Learning") and a group of 15 people face images to find that person (under the text 'Recognizing') where the correct recognization is labeled by "match" in green color and the wrong recognization has "ground-truth" and "predict" in red color.
<p align="center">
  <img src="https://raw.githubusercontent.com/DrMMZ/drmmz.github.io/master/images/celeba/2.JPG" width='380' height='420'/>
  <img src="https://raw.githubusercontent.com/DrMMZ/drmmz.github.io/master/images/celeba/celeba_movie.gif" width='380' height='420'/>
</p> 
The model is trained on the CelebA dataset following its default splitting with Adam optimizer for 60 epochs over 2 GPUs. It achieves the following result after 10 epochs on the test set where query examples in each episode contains exact 1 person from support example(s).

|3-shot|time (second)|mean (F1-score)|median (F1-score)|
|---|---|---|---|
|1-way, 15-query|0.04|0.91|1.0|
|1-way, 100-query|0.17|0.82|0.83|

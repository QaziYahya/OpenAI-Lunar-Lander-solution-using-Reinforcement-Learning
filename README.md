# OpenAI-Lunar-Lander-solution-using-Reinforcement-Learning
Reinforcement Learning DQN - using OpenAI Lunar Lander environment 

<li>Tensorflow</li>
<li>Keras</li>
<li>gym</li>
</br>

In the OpenAI Lunar Lander environment the goal is to successfully land a space ship on the moon, preferably on the landing pad represented by two flag poles.
The space ship can be controlled by using 4 discrete actions which are repersented by 0, 1, 2 and 3. They are described as follows:

<li>0: do nothing</li>
<li>1: fire left orientation engine</li>
<li>2: fire main engine</li>
<li>3: fire right orientation engine</li>
</br>

![Screenshot from 2022-09-24 20-39-04](https://user-images.githubusercontent.com/74414105/192106614-362a69b9-be98-498b-9b5d-750c0eb444d3.png)

The agent was trained using Deep-Q-Learning it took a total of 755 episodes for the model to train in about 25-30 minutes.

# Preview:

https://user-images.githubusercontent.com/74414105/192106742-c6360283-5888-457a-99d8-33136092aa96.mp4

# Training & Testing
The agent can be trained by running the "train.py" file keep in mind that it takes some time for the agent to train. The model can be tested by running the
"test.py" file it'll create and save a video of the agent playing the game. 

# Weather-Prediction

In this study, weather forecast operation was performed by using machine learning algorithms. First, machine learning methods were trained by using a dataset which was got from kaggle.com and then the system was tested with real data that come from sensors such as DHT11, BMP180. These sensors produce temperature, pressure and humidity real time data. The sensors were connected to Raspberry Pi device and both on laptop and R. Pi the necessary code implementation was completed. R. Pi was connected to laptop via ethernet cable and MobaXTerm third party software. It allowed to monitor program execution simultaneously.

### Dataset

Dataset which was used to train the machine learning models can be found in the following link: https://www.kaggle.com/muthuj7/weather-dataset. There was dramatic imbalance on dataset.

### The System

<p align="center">
  <img src="https://user-images.githubusercontent.com/57035819/150679594-e2d17e21-b87e-40c2-ba32-a6d0744fabb4.jpg"  width="700"/>
</p>

In the system, the following components were used to perform the the task:

- A DHT11 sensor for temperature and humidity
- A BMP180 for air pressure
- A 16x2 LCD Display to print the prediction for the user
- A buzzer and led to signal when the specific condition hits (in this case rainy output)

### Output and Success

The program output and the system's success details were mentioned below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/57035819/150679800-8c6037a8-8576-43c4-9ffc-621d939dfc82.png"  width="550"/>
  <img src="https://user-images.githubusercontent.com/57035819/150679818-92048238-7ed2-4c21-a617-1084888add45.png"  width="550"/>
</p>

In this study, temperature, humidity and air pressure values were used for training and prediction. In real life, there is more required information to achieve this. The information contains the followings:

- Apparent temperature
- Wind Speed
- Wind Bearing
- Visibility
- Loud Cover

### Author and Note
- [honourrable](https://github.com/honourrable)

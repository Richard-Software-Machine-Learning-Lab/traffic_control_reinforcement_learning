# Traffic Light Control with Reinforcement Learning

This project uses a Deep Q-Network (DQN) to optimize traffic signal control at a road intersection simulated in SUMO.


# Tech stack
* SUMO (Simulation of Urban MObility)
* TraCI (Traffic Control Interface)
* Python 3.0
* PyTorch
* Matplotlib


## Running the application 
```shell script
python3 main_training.py results_training 2 4
python3 nameFile.py folder experimentNumber inputNumber
python3 main_testing.py results_testing 2 80 80

```

## Key Features  

### Environment  
- 8-lane intersection with traffic lights using SUMO.  

### Traffic Signals  
- Two signal directions: **North-South** and **East-West**.  
- Green light duration: **11 seconds**.  
- Yellow light duration: **4 seconds**.  

### Traffic Flow  
- Simulations with **500** and **3,000+ vehicles**.  

### Reinforcement Learning Model  
- DQN used for traffic signal control to optimize policy learning.  

### State Representation  
- **Queue Length (QL)**: Number of vehicles waiting at each road.  
- **Vehicle Position (VP)**: Binary array representing vehicle presence in **80 cells**.  

### Actions  
- **Activate North-South** green light.  
- **Activate East-West** green light.  

### Reward  
- **Negative cumulative waiting time** of vehicles is minimized.  

### Training  
- **400 episodes** with **5,000 steps per episode** under low traffic conditions.  

### Evaluation  
- Performance compared between:  
  - **DQN Policy**  
  - **Random Policy**  
- Evaluated using cumulative vehicle waiting time.  

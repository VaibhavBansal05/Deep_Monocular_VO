| Model | Owner | ATE sim(3) [m] | ATE SE(3) [m] | RPE trans [m] | RPE rot [deg] | t_rel [%] | r_rel [deg/100m] | Params [M] | ms/frame |
|---|---|---|---|---|---|---|---|---|---|
| Classical ORB/LK + MLP | Pranav Joshi | **24.209** | **68.320** | 0.3878 | 0.2182 | 36.61 | 8.4002 | 0.27 | 9.7 |
| CNN + LSTM | Abhishek Laxman Jadhav | 61.646 | 82.835 | **0.3474** | 0.2508 | 54.52 | 19.7597 | 15.39 | 35.8 |
| Self-supervised PoseNet+DepthNet | Suryansh Bajaj | 25.082 | 227.123 | 0.9363 | **0.1883** | 97.39 | 8.7570 | 25.65 | 111.9 |
| CNN + CBAM attention | Vaibhav Krishna Bansal | 25.471 | 75.273 | 0.3637 | 0.1965 | **35.67** | **7.8879** | 11.23 | 17.1 |

> Scale handling differs by design: M1/M2/M4 learn metric translation from pose labels; self-supervised M3 has unresolved monocular scale. Its depth prediction does not set the trajectory's metric scale. Both ATE sim(3) (scale solved) and ATE SE(3) (scale fixed) are therefore reported. Bold marks the best value per column. Recorded ms/frame values came from different evaluation sessions and are not a matched speed comparison.
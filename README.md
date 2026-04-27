
# AIoT Acoustic Fault Detection Rover

## Overview
An ESP32-S3 based rover that detects machine faults 
in real time using acoustic signal analysis and edge AI.

## Hardware
- ESP32-S3 DevKit-C
- 2x INMP441 I2S microphones (noise cancellation)
- MPU6050 accelerometer
- L298N motor driver
- 2-wheel rover chassis

## Pin Map
| GPIO | Connected to        |
|------|---------------------|
| 4    | L298N ENA           |
| 5    | L298N ENB           |
| 6    | Red LED             |
| 8    | SDA (MPU6050)       |
| 9    | SCL (MPU6050)       |
| 11   | L298N IN1           |
| 12   | L298N IN2           |
| 13   | L298N IN3           |
| 14   | L298N IN4           |
| 15   | INMP441 WS          |
| 16   | INMP441 SCK         |
| 17   | INMP441 #1 SD       |
| 18   | Green LED           |
| 19   | Yellow LED          |
| 21   | INMP441 #2 SD       |

## ML Model
- Dataset: MIMII Fan (real factory sounds)
- Autoencoder: anomaly detection via reconstruction error
- Classifier: binary healthy vs anomaly
- Accuracy: 84.4% | False alarm rate: 4.9%
- Model size: 32.5 KB total

## How It Works
1. Two mics capture sound — mic2 subtracted from mic1 for noise cancellation
2. 68 acoustic features extracted per frame
3. Autoencoder computes health score 0-100
4. Classifier confirms fault type
5. Rover stops and RED LED fires on fault detection


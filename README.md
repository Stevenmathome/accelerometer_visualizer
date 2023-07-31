# accelerometer_visualizer
Using the Sparkfun OpenLog Artemis accelerometer chip data to visualize the x, y, and z movement in a gif

Uses Kalman filtering to ensure data noise is minimized

csv should be time,aX,aY,aZ formatted

Replace csv file on line 68

Rename file GIF saves as on line 110

Future iterations will include all 9-axis data to simulate all tilt, in hopes of simulating a firsbee's flight path and curve etc.

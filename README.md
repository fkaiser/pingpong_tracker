# PingPong_tracker
Tracks ball of ping pong via vision
## Selection of target histogram
The image below shows the histogram of the ball and some surrounding.   
![histogram](images_README/histogram.png)
## Computation of measurement update
Every update step of the particle filters involves the weighing of every single particle. This weighing is accomplished by computing the histogram within a rectangle around the particles center pointer and then comparing with the target histogram. As a comparison measure the so-called [**Hellinger distance**](https://en.wikipedia.org/wiki/Hellinger_distance) is selected. The **Hellinger distance** is related to the famous Bhattacharyya coefficient. The **Hellinger distance** is a value between 0 and 1. It takes 0 if there is perfect overlap and 1 if there is no overlap at all of the two distribution or this case discrete histograms. The figures below show how the value for the Hellinger distance behaves for the different pairwise fictional normal distribution. Both distribution have the same standard deviation and different average values. The bigger the average shifts the closer the Hellinger distance goes to 1.
![hellinger](images_README/hellinger_hist.png) 
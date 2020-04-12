# PingPong_tracker
The goal of the project to track a ping pong ball. The long term goal is to estimate the spin of the ball. To achieve those goals first and foremost one needs to be able to gather footage at high enough frequency and implement a procedure to track the ball. At the moment the footage is collected with a raspberry pi camera at a framerat of 80-90 Hz. For the tracking of the ball two different approches are given. On the one hand the ball is tracked via a particle filter framework more specifically via the [**Condensation algorithm**](https://en.wikipedia.org/wiki/Condensation_algorithm). On the other hand as a alternative the ball is tracked directly with the ciruclar hough transform. 

## Frames from video
The first important task to solve is to retrieve the "raw" pictures from a video e.g. from a .mp4. Additionally, it is extremely beneficial for the filter to know the exact timestamp of those pictures. To accomplish those tasks the functionality of ffmpeg and ffprobe under Linux are leveraged. If you execute the file `frames_from_video.sh` from the command line (make sure file is executable) all images of the video are extracted and stored as .png. On top of that the pts and dts timestamps are stored in the file timestamps.json. `pts` is an acronym for presentation time stamp and `dts` is an acronym for decoder time stamp. Both are extracted but actually only the pts timestamps are relevant later in the filter framework. To extract frames and timestamps of a video, type the following:

`./frames_from_video.sh <video> <folder to store frames and time stamps>`

## Installing python package
To run the motion tracker, install required python package by typing:
`pip install -r requirements.txt`

## Track the ball via particle filter
To start the tracker type:

`python motion_tracker.py --n_particle <desired number of particles> <path to folder with frames> --method 'particle'`

This will open a screen where you can select the region of interest. After pressing enter, the tracker will try to estimate the position of the selected object in the next frame. The region is then shown as green rectangle.

## Selection of target histogram
The image below shows the histogram of the ball and some surrounding.   
![histogram](images_README/histogram.png)
## Computation of measurement update
### Hellinger distance
Every update step of the particle filter procedure involves the weighing of every single particle. This weighing is accomplished by computing the histogram within a rectangle around the particles center point and then comparing with the target histogram. As a comparison measure the so-called [**Hellinger distance**](https://en.wikipedia.org/wiki/Hellinger_distance) is selected. The **Hellinger distance** is related to the famous Bhattacharyya coefficient and computed as follows:

<a href="https://www.codecogs.com/eqnedit.php?latex=d_{hellinger}(p^*,p(\vec{x_t}))&space;=&space;\sqrt{1&space;-&space;\sum_{j=1}^{m}&space;\sqrt{p_u^*(\vec{x_0})p_u(\vec{x_t})}}&space;\newline&space;\vec{x_0}:=&space;\text{States&space;of&space;target}&space;\newline&space;\vec{x_t}:=&space;\text{States&space;of&space;particle&space;t}&space;\newline&space;p_u^*(\vec{x_0}):=\text{Normalized&space;number&space;of&space;intensity&space;values&space;of&space;target&space;measurement&space;rectangle&space;that&space;lie&space;in&space;bin&space;\textit{u}}&space;\newline&space;p_u(\vec{x_t}):=\text{Normalized&space;number&space;of&space;intensity&space;values&space;of&space;particel&space;\textit{t}'s&space;measurement&space;rectangle&space;that&space;lie&space;in&space;bin&space;\textit{u}}&space;\newline&space;m:=\text{Number&space;of&space;bins&space;of&space;histograms}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d_{hellinger}(p^*,p(\vec{x_t}))&space;=&space;\sqrt{1&space;-&space;\sum_{j=1}^{m}&space;\sqrt{p_u^*(\vec{x_0})p_u(\vec{x_t})}}&space;\newline&space;\vec{x_0}:=&space;\text{States&space;of&space;target}&space;\newline&space;\vec{x_t}:=&space;\text{States&space;of&space;particle&space;t}&space;\newline&space;p_u^*(\vec{x_0}):=\text{Normalized&space;number&space;of&space;intensity&space;values&space;of&space;target&space;measurement&space;rectangle&space;that&space;lie&space;in&space;bin&space;\textit{u}}&space;\newline&space;p_u(\vec{x_t}):=\text{Normalized&space;number&space;of&space;intensity&space;values&space;of&space;particel&space;\textit{t}'s&space;measurement&space;rectangle&space;that&space;lie&space;in&space;bin&space;\textit{u}}&space;\newline&space;m:=\text{Number&space;of&space;bins&space;of&space;histograms}" title="d_{hellinger}(p^*,p(\vec{x_t})) = \sqrt{1 - \sum_{j=1}^{m} \sqrt{p_u^*(\vec{x_0})p_u(\vec{x_t})}} \newline \vec{x_0}:= \text{States of target} \newline \vec{x_t}:= \text{States of particle t} \newline p_u^*(\vec{x_0}):=\text{Normalized number of intensity values of target measurement rectangle that lie in bin \textit{u}} \newline p_u(\vec{x_t}):=\text{Normalized number of intensity values of particel \textit{t}'s measurement rectangle that lie in bin \textit{u}} \newline m:=\text{Number of bins of histograms}" /></a>

 The **Hellinger distance** takes values between 0 and 1. It takes 0 if there is perfect overlap and 1 if there is no overlap at all for the two distributions or in this case for the two discrete histograms. The figures below show how the value for the Hellinger distance behaves for the different pairwise fictional normal distribution. Both distributions have the same standard deviation but different mean values expect for the first plot where both entities are the same. The bigger the average shifts the closer the Hellinger distance goes to 1.
![hellinger](images_README/hellinger_hist.png)

If you want to play around with the different mean shift values or standard deviations or even different distribution like laplace you can do that by manipulating `distribution_comparison.py` accordingly. As mentioned in the equation the input for the computation of the Hellinger distance are the normalized bin counts i.e. 

<a href="https://www.codecogs.com/eqnedit.php?latex=p_u(\vec{x_t})=&space;\frac{n_u(\vec{x_t})}{\sum_{j=1}^m&space;n_j(\vec{x_t}))}\newline&space;n_u:=&space;\text{Bin&space;count&space;of&space;bin&space;\textit{u}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_u(\vec{x_t})=&space;\frac{n_u(\vec{x_t})}{\sum_{j=1}^m&space;n_j(\vec{x_t}))}\newline&space;n_u:=&space;\text{Bin&space;count&space;of&space;bin&space;\textit{u}}" title="p_u(\vec{x_t})= \frac{n_u(\vec{x_t})}{\sum_{j=1}^m n_j(\vec{x_t}))}\newline n_u:= \text{Bin count of bin \textit{u}}" /></a>

The following two plots show the difference of histograms of the cropped image of the target and a particle respectively the corresponding Hellinger distance value. As can be seen in the first example below even if the ball is almost not part of the cropped image anymore the value of the Hellinger distance is still below 0.5. This is due to the fact that the background on both images is still the same or really similar. The second image shows a particle where the ball is in the cropped image with a major part. Thus the Hellinger distance is smaller.
![half_ball](images_README/half_ball.png)
![complete_ball](images_README/complete_ball.png)

### Importance sampling
Obviously the question arises how to use the computed Hellinger distance to compute the posterior estimate of the position of the ping pong ball. Ideally, one would know the posterior distribution of p(x|z). Unfortunately, this is not the case here i.e. one does not know how exactly the different Hellinger distances are statistically distributed. To still be able to approximate the posterior distribution a so-called biased distribution q(x|z) is introduced. We assume that the posterior distribution is proportional to the biased distribution i.e.

<a href="https://www.codecogs.com/eqnedit.php?latex=w_k\propto&space;\frac{p(x_k|z)}{q(x_k|z)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_k\propto&space;\frac{p(x_k|z)}{q(x_k|z)}" title="w_k\propto \frac{p(x_k|z)}{q(x_k|z)}" /></a>

This also means that if we re-sample our prior estimated particles according to the biased distribution one can approximate the posterior distribution p(x|z) since samples with high weights are chosen many times. For the biased distribution q(x|z) it is assumed that it is the normal distribution with zero mean and a specific variance:

<a href="https://www.codecogs.com/eqnedit.php?latex=q(d_{hellinger})&space;=&space;\frac{1}{\sqrt{2\pi\sigma^2}}exp(-\frac{d_{hellinger}^2}{2\sigma^2})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?q(d_{hellinger})&space;=&space;\frac{1}{\sqrt{2\pi\sigma^2}}exp(-\frac{d_{hellinger}^2}{2\sigma^2})" title="q(d_{hellinger}) = \frac{1}{\sqrt{2\pi\sigma^2}}exp(-\frac{d_{hellinger}^2}{2\sigma^2})" /></a>

### Estimation of the ball
The effective position of the rectangel containing the ball is the minimum mean square error (MMSE) estimator that this estimate minimize the mean squared error to the true center position of the rectangle containing the ball. 

<a href="https://www.codecogs.com/eqnedit.php?latex=\vec{x}_{MMSE}&space;=&space;\frac{1}{n}\sum_{i=1}^{n}&space;\vec{x}_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\vec{x}_{MMSE}&space;=&space;\frac{1}{n}\sum_{i=1}^{n}&space;\vec{x}_i" title="\vec{x}_{MMSE} = \frac{1}{n}\sum_{i=1}^{n} \vec{x}_i" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\vec{x}_i&space;:=&space;\text{center&space;position&space;of&space;retangle&space;i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\vec{x}_i&space;:=&space;\text{center&space;position&space;of&space;retangle&space;i}" title="\vec{x}_i := \text{center position of retangle i}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=n:=&space;\text{Total&space;number&space;of&space;rectangles&space;in&space;filter}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n:=&space;\text{Total&space;number&space;of&space;rectangles&space;in&space;filter}" title="n:= \text{Total number of rectangles in filter}" /></a>

### Result of tracker
As an example of the tracker a black ball is used as a pendulum. The tracker tries to track the position of the ball. In the image below you see all rectangles of the filter with a blue border respectively the MMSE estimate has a green border. The red circle(s) represent the centers of the rectangles after resampling. In the image below there are only six center(s). This is done due to resampling to account for the higher probability that the ball is in this area. In order to avoid a situation where the filter has only one particle at every propagation step there is some noise applied. Hence, filter is more robust against fast motions.

![pendulum](images_README/pend_tracker.png)

The correspoding video can be found in:

![pendulum_ball](images_README/pendulum_ball.mp4)

## Track the ball via circular hough transform
The [circular hough transform](https://en.wikipedia.org/wiki/Circle_Hough_Transform) tries to extract circules in imagery by process the image in a way that allows to associates circles in a image to an accumulator space that represents the 3D parameter space for circles. By doing so one can than effevictely where the images are localized. Generally the steps of the circular hough transform involve the following:
* Reduce noise on image e.g. with Gaussian or median blurring
* Convert to gray scale image
* Apply the Canny-edge detector
* Vote the possible circles in the accumlator space
* Local maxima in the accumlator space represent circles in the image

OpenCV offers a direct implementation of the circular hough transform described in more detail [here](https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_circle/hough_circle.html).

To start the tracker type:

`python motion_tracker.py  <path to folder with frames> --method 'hough'`

As an exmaple of the circular hough tracker in the image below the green circle respectivly the red dot show the estimated circle respectively the mid point of the circle.

![hough](images_README/hough_tracker.png)

## Creating video from stored processed frames
You can select the option to store the processed frames similar to the image shown in the last section. If you want to create a video based on those frames, you can use the script video_from_frame.sh by typing:

`./video_from_frame.sh <image folder> <desired frame rate e.g. 30>`

The video is then stored in the processed folder with the name output.mp4.



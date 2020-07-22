# puzzle match
The program uses sift as basic feature matching method, and uses two methods for calculating the correct R and T to move puzzle.
## method 1 
Calculate the homography matrix, and decompose it.
## method 2
Calculate the R and T through RANSAC, and optimize it by nonlinear LSE to calculate an accurate result.
## result
[!img](../results/fig1.jpg)
[!img](../results/fig2.jpg)

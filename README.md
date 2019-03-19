# EDCircles
[EDCircles: A real-time circle detector with a false detection control](https://www.sciencedirect.com/science/article/abs/pii/S0031320312004268)

### Disclaimer
I have tried to implement EDCircles as close as what I have drawn from the paper. However, I am not responsible for any issues that could happen due to the use of this project.

### EDCircles flow
<img alt="02" src="https://user-images.githubusercontent.com/16577855/53748054-e90fb900-3ee7-11e9-85d8-3dda36654b18.png">

### Dependencies
+ OpenCV 4.0.0

### Build & Run
+ Build
```
make all
```
+ Clean
```
make clean
```
+ Run
```
./edcircles path/to/image
```

### Some results
+ Source image
<img alt="02" src="https://user-images.githubusercontent.com/16577855/54619430-ebf1d880-4aa7-11e9-9971-ad51f44906a5.png">

+ Edge segments
<img alt="02" src="https://user-images.githubusercontent.com/16577855/54619446-f1e7b980-4aa7-11e9-8e9b-824bf0ca79d2.png">

+ Circle candidates
<img alt="02" src="https://user-images.githubusercontent.com/16577855/54619474-01670280-4aa8-11e9-96f5-4b2d56a0d6f3.png">

+ Line candidates
<img alt="02" src="https://user-images.githubusercontent.com/16577855/54619491-06c44d00-4aa8-11e9-9d19-2e25a7448453.png">

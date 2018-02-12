# NFL Game Simulator

Studied and implemented the coordinate descent algorithm to fit a model of NFL team offensive and defensive quality. The model is trained on data from a collection of games.

Implemented code to test whether a denser or a sparse model gives the best predictions for the 7th week in the 2018 season. Determined that coordinate descent becomes slower with smaller penalties. We can conclude that because smaller penalties result in larger values of the loss function. Therefore, when we run ternary search, it takes longer for the model to converge.

Also implemented the function to determine the optimal penalty from to get a model with a desired number of parameters.

![Number of Non Zero Parameters v. Magnitude of Test Error](https://github.com/akkarh/HW2/blob/master/plot.png "Number of Non Zero Parameters v. Magnitude of Test Error")


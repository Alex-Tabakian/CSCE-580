Part A. Data Collection [5 pt]

    1. Model Building and Sanity Checking [15 pt]
    Part (a) Convolutional Network - 5 pt
    Build a convolutional neural network model that takes the (224x224 RGB) image as input, and predicts the letter. Your model should be a subclass of nn.Module. Explain your choice of neural network architecture: how many layers did you choose? What types of layers did you use? Were they fully-connected or convolutional? What about other decisions like pooling layers, activation functions, number of channels / hidden units?


    Part (b) Training Code - 5 pt
    Write code that trains your neural network given some training data. Your training code should make it easy to tweak the usual hyperparameters, like batch size, learning rate, and the model object itself. Make sure that you are checkpointing your models from time to time (the frequency is up to you). Explain your choice of loss function and optimizer.

    Part (c) â€œOverfitâ€ to a Small Dataset - 5 pt
    One way to sanity check our neural network model and training code is to check whether the model is capable of â€œoverfittingâ€ or â€œmemorizingâ€ a small dataset. A properly constructed CNN with correct training code should be able to memorize the answers to a small number of images quickly.

    Construct a small dataset (e.g.Â just the images that you have collected). Then show that your model and training code is capable of memorizing the labels of this small data set.

    With a large batch size (e.g.Â the entire small dataset) and learning rate that is not too high, You should be able to obtain a 100% training accuracy on that small dataset relatively quickly (within 200 iterations).


Part B. Building a CNN [35 pt]

    1. Model Building and Sanity Checking [15 pt]

        Part (a) Convolutional Network - 5 pt
        Build a convolutional neural network model that takes the (224x224 RGB) image as input, and predicts the letter. Your model should be a subclass of nn.Module. Explain your choice of neural network architecture: how many layers did you choose? What types of layers did you use? Were they fully-connected or convolutional? What about other decisions like pooling layers, activation functions, number of channels / hidden units?
            There are 3 convolutional layers, 2 fully connected layers, 3 pooling layers, and 4 ReLU activation layers. 

        Part (b) Training Code - 5 pt
        Write code that trains your neural network given some training data. Your training code should make it easy to tweak the usual hyperparameters, like batch size, learning rate, and the model object itself. Make sure that you are checkpointing your models from time to time (the frequency is up to you). Explain your choice of loss function and optimizer.

        Part (c) â€œOverfitâ€ to a Small Dataset - 5 pt
        One way to sanity check our neural network model and training code is to check whether the model is capable of â€œoverfittingâ€ or â€œmemorizingâ€ a small dataset. A properly constructed CNN with correct training code should be able to memorize the answers to a small number of images quickly.

    Construct a small dataset (e.g.Â just the images that you have collected). Then show that your model and training code is capable of memorizing the labels of this small data set.

    With a large batch size (e.g.Â the entire small dataset) and learning rate that is not too high, You should be able to obtain a 100% training accuracy on that small dataset relatively quickly (within 200 iterations).

    2. Data Loading and Splitting [5 pt]
    Download the anonymized data collected by you and your classmates. Split the data into training, validation, and test sets.

    Note: Data splitting is not as trivial in this lab. We want our test set to closely resemble the setting in which our model will be used. In particular, our test set should contain hands that are never seen in training! Remember that test sets are used to estimate how well the model will generalize to new data, and that â€œnew dataâ€ will involve hands that the model has never seen before.

    Explain how you split the data, either by describing what you did, or by showing the code that you used. Justify your choice of splitting strategy. How many training, validation, and test images do you have?

    For loading the data, you can use plt.imread as in Lab 1, or any other method that you choose. You may find torchvision.datasets.ImageFolder helpful. (see https://pytorch.org/docs/master/torchvision/datasets.html#imagefolder ) For this portion only, you are free to look up tutorials or other code on the internet to help you.

    3. Training [5 pt]
    Train your first network on your training set. Plot the training curve, and include your plot in your writeup. Make sure that you are checkpointing frequently!

    4. Hyperparameter Search [10 pt]
        Part (a) - 1 pt
        List 3 hyperparameters that you think are most worth tuning. Choose at least one hyperparameter related to the model architecture.
            Three hyperparameters that I could change are learning rate(lr), batch size, and number of channels. I will be modifying the learning rate and batch to try to get the parameters for the best AI model


        Part (b) - 6 pt
        Tune the hyperparameters you listed in Part (a), trying as many values as you need to until you feel satisfied that you are getting a good model. Plot the training curve of at least 4 different hyperparameter settings.
            I trained the models with four different hyperparameter settings. I used differant learning rates and batch sizes for each model. The model uses 
            20 epochs are run with 6,700 images (10% of total image set)
            1. batch size = 16, learning rate = 1e-3: Final validation accuracy = 89.43%
            2. batch size = 32, learning rate = 1e-3: Final validation accuracy = 85.86%
            3. batch size = 16, learning rate = 1e-4: Final validation accuracy = 86.78%
            4. batch size = 32, learning rate = 1e-3: Final validation accuracy = 89.66%

        Part (c) - 1 pt
        Choose the best model out of all the ones that you have trained. Justify your choice.
            Model number 4 with a batch size of 32 and a learning rate of 1e-3 preformed the best since it had the highest valudation accuracy.
        Part (d) - 2 pt
        Report the test accuracy of your best model. You should only do this step once.

Part C. Transfer Learning [15 pt]
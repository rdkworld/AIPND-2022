
# Project 1 - Use a Pre-trained Image Classifier to Identify Dog Breeds
This project will implement Python code to use a trained image classifier to identify dog breeds.

## Principal Objectives

1. Correctly identify which pet images are of dogs (even if the breed is misclassified) and which pet images aren't of dogs.
2. Correctly classify the breed of dog, for the images that are of dogs.
3. Determine which CNN model architecture (**ResNet, AlexNet, or VGG**), "best" achieve objectives 1 and 2.
4. Consider the time resources required to best achieve objectives 1 and 2, and determine if an alternative solution would have given a "good enough" result, given the amount of time each of the algorithms takes to run.
 
## Program Outline

- Time your program
    - Use Time Module to compute program runtime
- Get program Inputs from the user
    - Use command line arguments to get user inputs
- Create Pet Images Labels
    - Use the pet images filenames to create labels
    - Store the pet image labels in a data structure (e.g. dictionary)
- Create Classifier Labels and Compare Labels
    - Use the Classifier function to classify the images and create the classifier labels
    - Compare Classifier Labels to Pet Image Labels
    - Store Pet Labels, Classifier Labels, and their comparison in a complex data structure (e.g. dictionary of lists)
- Classifying Labels as "Dogs" or "Not Dogs"
    - Classify all Labels as "Dogs" or "Not Dogs" using dognames.txt file
    - Store new classifications in the complex data structure (e.g. dictionary of lists)
- Calculate the Results
    - Use Labels and their classifications to determine how well the algorithm worked on classifying images
- Print the Results

Above tasks will be repeated for each of the three image classification algorithms.

## Final Results

| Type  | Count |
| ------------- | ------------- |
| # Total Images  | 40  |
| # Dog Images  | 30  |
| # Not-a-Dog Images  | 10  |

| CNN Model Architecture  | % Not-a-Dog Correct | % Dogs Correct | % Breeds Correct | % Match Labels |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| **ResNet**  | 90.0%  | 100.0%  | 90.0%  | 82.5%  |
| **AlexNet**  | 100.0%  | 100.0%  | 80.0%  | 75.0%  |
| **VGG**  | 100.0%  | 100.0%  | 93.3%  | 87.5%  |

**Conclusion**

Given above results, the **"best"** model architecture is **VGG**. It outperformed both of the other architectures when considering principal objectives 1 and 2. We noticed that ResNet did classify dog breeds better than AlexNet, but only VGG and AlexNet were able to classify "dogs" and "not-a-dog" at 100% accuracy. The model VGG was the one that was able to classify "dogs" and "not-a-dog" with 100% accuracy and had the best performance regarding breed classification with 93.3% accuracy.

## Source Code

All the source code is available under **Project 1** folder including input sample images.

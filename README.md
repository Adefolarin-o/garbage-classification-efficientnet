# **Garbage Classification using EfficientNetV2B0 by Group 5**

## **Project Overview**

This project builds a deep learning pipeline for classifying garbage images into different categories. It leverages transfer learning with the **EfficientNetV2B0** model, combined with data augmentation and a two-phase training strategy (initial training with a frozen base followed by fine-tuning). An interactive test explorer is included to visualize predictions on test images.

## **Table of Contents**

* Installation  
* Dataset Preparation  
* Project Structure  
* Model Training and Fine-tuning  
* Evaluation & Model Saving  
* Interactive Test Explorer  
* Usage Instructions  
* Credits  
* License

## **Installation**

This project requires Python 3.x. Install the following dependencies:

`pip install kagglehub tensorflow keras keras_cv matplotlib ipywidgets`

Ensure that you are running the code in an environment with GPU support (e.g., Google Colab with GPU enabled) for faster training.

## **Dataset Preparation**

* **Downloading the Dataset:**  
  The dataset is automatically downloaded from Kaggle using `kagglehub` (dataset: `sumn2u/garbage-classification-v2`).  
* **Data Splitting:**  
  The code splits the dataset into training (70%), validation (20%), and test (10%) directories. It shuffles images for each class and copies them into their respective folders.

## **Project Structure**

The code is organized into the following cells:

1. **Setup & Dependencies:**  
   * Installs required packages.  
   * Sets up mixed precision for faster training.  
2. **Dataset Download & Preparation:**  
   * Downloads the dataset using *`kagglehub`*.  
   * Prepares and splits the dataset into training, validation, and test sets.  
3. **Data Splitting & Verification:**  
   * Contains a function to split the dataset by class.  
   * Verifies the distribution of images across splits.  
4. **Data Preprocessing & Generators:**  
   * Uses *`ImageDataGenerator`* for data augmentation and preprocessing (with EfficientNet’s *`preprocess_input`*).  
   * Sets up data generators for training, validation, and testing.  
5. **Model Building:**  
   * Constructs the model using EfficientNetV2B0 as a frozen base.  
   * Adds a custom classification head with a dense layer, dropout, and final softmax output.  
6. **Training – Phase 1 (Frozen Base):**  
   * Trains the model for 20 epochs with early stopping and learning rate reduction callbacks.  
7. **Training – Phase 2 (Fine-Tuning):**  
   * Unfreezes the top 20% of the base model layers.  
   * Re-compiles and trains the model for 15 additional epochs with adjusted learning rate and callbacks.  
8. **Evaluation & Saving the Model:**  
   * Evaluates the model on the test set.  
   * Saves the trained model in Keras v3 format with a `.keras` extension.  
   * Demonstrates how to load the saved model.  
9. **Interactive Test Explorer:**  
   * Provides an interactive widget interface to select test images.  
   * Displays the selected image along with its true label, predicted label, and a confidence bar chart for all classes.

## **Model Training and Fine-Tuning**

* **Phase 1:**  
  The EfficientNetV2B0 base model is loaded with frozen weights. A custom head is added, and the model is trained for 20 epochs using early stopping and learning rate reduction callbacks.  
* **Phase 2:**  
  The top 20% of the base layers is unfrozen for fine-tuning over 15 additional epochs, with a lower learning rate to adapt the model to the garbage dataset.


## **Evaluation and Model Saving**

* **Evaluation:**  
  The model is evaluated on the test set to report accuracy and loss.

**Saving the Model:**  
The final model is saved in the modern `.keras` format to Google Drive.  
**Example command in the notebook:**  
*`model.save("/content/drive/MyDrive/garbage_classifier_v2.keras")`*

*   
* **Verification:**  
  Additional cells confirm that the model file exists and demonstrate how to load it.

## **Interactive Test Explorer**

An interactive widget-based test explorer (using IPyWidgets and Matplotlib) is provided to:

* **Select a Test Sample:**  
  Use a dropdown menu to choose a test image.  
* **Adjust Confidence Threshold:**  
  Use a slider to set the threshold for prediction confidence.  
* **Visualize Predictions:**  
  Display the chosen image, its true label, the predicted label with confidence, and a bar chart showing the model’s confidence distribution across all classes.

## **Usage Instructions**

* **Installation:** Follow the installation instructions above.  
* **Running the Notebook:** Open the notebook in Google Colab.  
* **Downloading the Model:** Download the model shared via google drive named “*garbage\_classifier\_v2.keras*”, and add it to your Google Drive’s “*My Drive*” directory.  
* **Executing Cells:**  
  Run cells in the specified order (Cells 1–5, then 10, 12, 13, and finally 14).  
* **Exploring Predictions:**  
  Use the Interactive Test Explorer (Cell 14) to view sample predictions, adjust confidence thresholds, and visualize the output.

## **Credits**

* **Developed by:**  
  * Adefolarin Olateru-Olagbegi  
  * Jahir Bakari  
  * Alexander Coughlon  
  * Jamabo William-West  
* **Dataset:**  
  Garbage Classification v2 on Kaggle  
* **Technologies:**  
  TensorFlow, Keras, EfficientNetV2B0, IPyWidgets

## **References**

* [https://www.tensorflow.org/api\_docs/python/tf/keras/preprocessing/image/ImageDataGenerator?utm\_source=chatgpt.com](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator?utm_source=chatgpt.com)  
* [https://keras.io/guides/transfer\_learning/](https://keras.io/guides/transfer_learning/)  
* [https://www.datacamp.com/tutorial/complete-guide-data-augmentation?utm\_source=chatgpt.com](https://www.datacamp.com/tutorial/complete-guide-data-augmentation?utm_source=chatgpt.com)  
* [https://keras.io/api/applications/efficientnet\_v2/](https://keras.io/api/applications/efficientnet_v2/)  
* [https://www.tensorflow.org/api\_docs/python/tf/keras/layers/GlobalAveragePooling2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling2D)  
  


  

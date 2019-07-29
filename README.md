# Breast Cancer Classification
A Convolutional Neural Network which implements transfer learning from the VGG19 model to predict whether or not a patient has invasive ductal carcinoma (IDC), a leading form of breast cancer, based on an input image.

Breast cancer is the most common form of cancer in women, and IDC is the most common form of breast cancer. Accurately identifying and categorizing breast cancer subtypes is an important clinical task, and automated methods can be used to save time and reduce error.

The data has been derived from a Kaggle dataset, which is itself derived from numerous medical journals.

The model achieves an overall test-set accuracy of approximately 90%

### Required Dependencies
* Numpy
* Pandas
* Matplotlib
* Ski Kit Learn
* Keras
* Random
* OpenCV2


### Execution Instructions
1. Download the raw dataset zip file from the following Kaggle link: https://www.kaggle.com/paultimothymooney/breast-histopathology-images

2. Configure all constants in project.
~~~~
python config.py
~~~~~~~~ 

3. Unzip and organise downloaded dataset.
~~~~
python setup.py
~~~~~~~~ 

4. Organise and structure dataset.
~~~~
python preprocess.py
~~~~~~~~ 

5. Create model class.
~~~~
python model.py
~~~~~~~~ 

6. Train model.
~~~~
python train.py
~~~~~~~~ 

7. Test model, substituting 'imageName' for the name of your test file.
~~~~
python predict.py -i <imageName.jpg>
~~~~~~~~ 

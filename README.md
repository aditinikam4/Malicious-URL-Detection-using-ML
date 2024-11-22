### Malicious URL Detection Using Machine Learning
This project detects and classifies URLs into categories such as Benign, Defacement, Phishing, or Malware using various machine learning classifiers. The model is trained to identify whether a URL is safe or potentially harmful based on extracted features.

### Libraries Used:
This project utilizes the following Python libraries:
1. Pandas: For data manipulation and analysis.
2. Scikit-learn: For machine learning algorithms and evaluation metrics.
3. Pickle: For model serialization and deserialization.
4. Numpy: For numerical computations.
5. Requests: To fetch URLs for prediction.

### Project Workflow:
1. Data Collection:
The dataset contains URLs labeled as Benign, Defacement, Phishing, or Malware.
Features are extracted from the URLs, such as domain components, length, presence of HTTPS, and other relevant indicators.
2. Feature Engineering:
URL components are preprocessed and transformed into numerical features that can be used by machine learning models.
3. Model Training:
The following classifiers are used for training:
AdaBoost Classifier
Random Forest Classifier
The models are trained on the dataset and evaluated using accuracy, precision, recall, and other relevant metrics.
4. Model Evaluation:
The performance of each model is measured by calculating Accuracy, Precision, Recall, and F1-Score for each category (Benign, Defacement, Phishing, Malware).
A confusion matrix is also generated to visualize how well the models classify URLs into these categories.
5. Model Serialization:
The trained models are saved using the pickle module so that they can be reused for future predictions without retraining.
6. Prediction:
After training, the model can predict whether a URL is Benign, Defacement, Phishing, or Malware by inputting the URL for classification.
7. Accuracy & Precision
The model's performance was evaluated using various metrics:
Accuracy: The proportion of correct predictions.
Precision: The ratio of true positive predictions to all positive predictions.
Recall: The ratio of true positives to all actual positive cases.
F1-Score: A balance between precision and recall.

### Example Evaluation Results:
Model	Accuracy (%)	Precision (%)	Recall (%)	F1-Score (%)
AdaBoost Classifier	94.2	92.6	95.4	94.0
Random Forest	96.8	94.2	98.0	96.1

### Model Classification Details:
Benign: URLs that are safe and not harmful.
Defacement: URLs associated with website defacement, typically showing altered content.
Phishing: URLs used to impersonate legitimate websites for fraudulent activities.
Malware: URLs linked to websites that distribute harmful software.

### How to Use:
Here’s the "How to Use" section with short steps, without the code:

---

## How to Use

### 1. Download the Dataset-
- Download the `malicious_phish.csv` dataset from Kaggle. Search for the dataset on Kaggle and click the Download button. Link- https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset?resource=download

### 2. Install Required Libraries:
- Install all necessary Python libraries by running:
  ```bash
  pip install -r requirements.txt
  ```

### 3. Run the Code Step by Step:
Follow these steps to run the project:
## Step 1: Import the Dataset
- Load the `malicious_phish.csv` file into your environment and inspect the first few rows.
## Step 2: Data Exploration
- Check the distribution of URLs across categories (Benign, Defacement, Phishing, Malware).
## Step 3: Calculate URL Metrics:
- Calculate the percentage of URLs that use HTTPS.
- Identify and calculate the percentage of shortened URLs.
- Check for URLs that use IP addresses instead of domain names.
## Step 4: Train the Model:
- Split the dataset into training and testing sets.
- Train the model using classifiers like AdaBoostand Random Forest.
## Step 5: Evaluate the Model:
- Evaluate the models based on accuracy, precision, recall, and F1-score.
## Step 6: Save the Model:
- Save the trained model using Pickle for future use.
## Step 7: Predict URL Safety
- Load the saved model and input URLs to predict whether they are **Safe** or **Unsafe**.

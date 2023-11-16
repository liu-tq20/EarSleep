# EarSleep

Hearing your sleep stage



### Data Collection


Data from a single night of sleep for five individuals was obtained, using an Android phone accelerometer for motion data and a Huawei Watch 2 for heart rate data. The data was saved as CSV files, which are located in the `./data` folder for analysis. Sleep stages—awake, deep (REM, NREM3), and light sleep (NREM1, NREM2)—were categorized using an established sleep app.

Referred to this [repository](https://github.com/bpine0/Sleep-Stages-Classification).


### Model Training

`Model_train.py` outlines the training process. Data from all CSVs were compiled and synchronized by timestamps. Features were then extracted from the acceleration and heart rate datasets. A random forest classifier was employed for classification purposes.



### Recognizing sleep stage

The `Model_recognition.py` script uses the pre-trained model to categorize raw sleep data into stages.



### Three stage classification

Key metrics for the three-stage classification are as follows:

- Average overall accuracy: 98.6%
- Awake class accuracy: 99.9%
- Awake precision: 98.1%
- Awake recall: 93.5%
- Light sleep class accuracy:98.9%
- Light sleep precision: 93.8%
- Light sleep recall: 96.7%
- Deep sleep class accuracy: 97.1%
- Deep sleep precision: 93.6%
- Deep sleep recall: 88.2%

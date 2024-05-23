># **HOW I FOUND THE DATASET** 
>## **Dataset Source**:
>- **Initial Search**: I searched for a dataset relating lung capacity to various factors, but I couldn't find the exact dataset required for the task.
>- **Alternative Dataset**: Eventually, I found a suitable dataset on Kaggle that includes the following features: Age (years), Height (inches), Smoke, Gender, Caesarean, Number of Children, and Weight (kg). This dataset contains 746 entries.

>## **Deriving Tidal Volume**:
>- **Research Reference**: I referred to an article from the American Thoracic Society Journals that discusses the relationship between tidal volume and the aforementioned factors. The article is accessible here.
https://www.atsjournals.org/doi/full/10.1164/rccm.200901-0127oc

>- **Tidal Volume Calculation**: Based on the formulas provided in the article, I calculated tidal volume for men and women separately:
    >- for men :  (23.79 + 0.00002066 * age_cubed) * row['LungCap(cc)'] / 100
    >- for women : (27.091 + 0.00002554 * age_cubed) * row['LungCap(cc)'] / 100
    


># MY APPROACH TO THE PS 
### Data Preparation Steps:
1. **Loading Data**: Loaded the dataset from a CSV file.
2. **Column Renaming**: Renamed columns to more convenient names.
3. **Handling Missing Values**:
   - Filled missing values in numeric columns with the mean of each column.
   - Filled missing values in categorical columns with the mode of each column.
4. **Feature Engineering**: 
   - Calculated the tidal volume for each entry based on the gender and age.
   - Added the calculated tidal volume as a new column in the dataset.
5. **Sorting Data**: Sorted the dataset by the 'Age' column in ascending order.
6. **Saving Processed Data**: Saved the cleaned and sorted dataset to a new CSV file named 'cleaned_sorted_data.csv'.
### Pre-processing the data:
1. **Define Predictor and Target Variables**:
   -Predictors (X): Age, Gender, Height, Smoke, Caesarean
   -Target (y): Tidal Volume

2. **Preprocess the Data**:
Numeric Features: Scale using StandardScaler.
Categorical Features: One-hot encode using OneHotEncoder.
3. **Split the Data**: Use train_test_split to split the data into training and testing sets.

### Defining,compiling,trainig the neural network 

### Evaluating the neural network beofore tuning the hyperparameters 

### Evaluating the neural network after finding the best fit for hyperparameters 


# Problems I Faced

### Unicode Error

- I did some general debugging after which I kept encountering a Unicode error regarding U+2010, which is a hyphen.
- I checked both the lungcapacity.csv file and cleaned_sorted_data.csv and could not find any hyphen.
- I thought maybe it was mistaking the negative sign introduced in the file due to preprocessing, which reduces the value for easy processing. Even after changing the numeric transformer from StandardScaler to MinMaxScaler, which gives only positive output values, the error didn't go away.
- I further introduced `utf-8` as encoding in every function in case there was some conversion. I have attached a screenshot of my code at that time. However, the error did not go away.
- I even went to the library definition of the function and thought of altering that, introducing a different encoding or such, but that didn't work out.
- After wasting more than an hour on this, I found this wonderful website which saved me: [GitHub Issue #19386](https://github.com/keras-team/keras/issues/19386), which made me realize the problem is probably in Windows 11 itself.
- Then, after some research, I found out the magic of setting `verbose=0`, which stops the progress bar in Windows 11 from initiating during every epoch, where the problem was coming from.
- It was a problem in the Windows 11 system.

### Miscellaneous

- Except for this, I didn't really encounter anything very interesting that good old definitions of the functions in the imported libraries couldn't solve. However, I guess it's worth mentioning that the first response of AI tools when asked about a problem is to vectorize the X and y datasets, which makes it, needless to say, into a hydra's head and causes exponential growth in errors , because it then forces you to keep playing around with the format of the data , from numpy arrays, tensors to subsets with half the required attributes not being defined for the type the data is stored in and the functions simply rejecting the format the input data is in . All of this makes the idea of throwing your laptop on the wall very appealing , but all you have to do is .... completely start again . 

### Hyperparameter tuning
-I was convinced I wrote the correct code importing one of the best models , Bayesian , however it kept getting stuck when the function was called , the line where estimator kept giving me errors regardless of what I did and even after hours , I couldnt find the answer  . 
- I am pretty sure the rest of the commented out code is correct as it is simply a repeat of before .
-So I leave it commented for now , but I will keep looking for the answer even after submission 


># HOW I INCREASED ACCURACY 
>- I calculated several metrics on which the data could be judged and random.seed ensured that I got consistent accuracy measurements 

># HOW I CONVERTED SKLEARN FUNCTIONS INTO FUNCTIONS OF PYTORCH AND TENSORFLOW 
>### accuracy function in sklearn 
-I replaced with its actual mathematical interpretation after research on the internet 
 rather than accuracy_score(y_pred,y_test) , I used
'Accuracy: %d' % float((np.dot(y_test_classes,y_pred_classes.T) + np.dot(1-y_test_classes,1-y_pred_classes.T))/float(y_test_classes.size)*100) + '%'

>### train_test_split() function 
-I tried to split the datasets in a particular ratio , but I kept getting the error ValueError: could not convert string to float: 'yes', even though I had performed categorical transform in preprocessing and it should have worked.Could not find a solution even after extensive research on the Internet and thus had to stick with the train_test_split() function . 
- Before this I also tried several other ideas like , 
- torch.utils.data.random_split() on the X and y datasets I converted to tensors and then for X as it was giving error in being processed as it was a mix of strings and ints , I enumerated through different columns , concatenated the strings and made it different from the numerical part , but it didnt work .
- I tried to split the data into train and test sets using DataLoader and extract data and targets from the DataLoader datasets , but that didnt work either 

>-Tried to define preprocessing steps , trying to replace standarrdencoder and onehotencoder by dividing by decimal values and implementing some complex mathematics functions , but that gave error because of the strings and standardencoder is still easy to visualise but onehotencoder , I could not do only in pytorch and tensorflow . 
>- Found some really complex code claiming it worked without sklearn , but it sadly didnt work at all https://www.analyticsvidhya.com/blog/2021/05/tuning-the-hyperparameters-and-layers-of-neural-network-deep-learning/
>-Imported several many libraries in the hope I will be able to replace sklearn however failed due to some pesky errors , but I leave most libraries still imported , all those packages that took hours to install in my pc , because it makes me very sad to delete them . They dont bother the code as such , just increase runtime by less than a second .
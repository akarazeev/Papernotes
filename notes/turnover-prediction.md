### Employee Turnover Prediction

* Bank of USA
  * 2013-2016
  * 28% had left
  * 14k employee entries
  * 24 features
* IBM Watson Analytics
  * 16% had left
  * 1470 employee entries
  * 38 features

HR features: age, compensations, gender, education, ...

#### Preprocessing

* Missing values
  * numeric type replaced with median value
  * categorical type replaced with mode value
* Data type Conversion
  * categorical -> numeric
  * OHE
  * label encoding via sklearn
* Feature Selection: PCA to reduce dimensionality (wasn't used)
* Feature Scaling: scale the inputs to achieve good results (normalization and standardization were performed)

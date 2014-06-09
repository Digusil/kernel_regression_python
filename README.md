# Kernel Regression

A tiny collection of functions and classes to do some machine learning research on multidimensional datasets based on kernel regression.

## Why another regression system

Regression systems are usable tools to analyze datasets which concludes inputs and targets. Some of these kind of machine learning systems are based on hypothesis. The success is bounded to the quality of the hypothesis. Often it is practical to define a specific hypothesis function with particular features, so that the dataset can be analyzed on this features. In other cases the regression model should fit the dataset as good as possible. In this case the hypothesis should *only* fit the data. An automatically generated hypothesis is usable, if no special features are needed. 

## What does my kernel_regression

Kernel Regression is a nonparametric regression concept that produces its own hypothesis. The given feature tuples {x, y} will be used to generate the hypothesis. A kernel function K(u) evaluates the significance of the several feature points. The hypothesis will be calculated based on the Nadaraya-Watson-Estimator-Concept m_i = sum(y_j Kh(u_ij))/sum(Kh(u_ij)). As cost function the mean-squared-error (MSE) is implemented.

My kernel regression supports different modes for parameterizing the kernel function. Its possible to use a general bandwidth h over all feature points or to optimize separately for each. Also you can choose if you want to use a scaled Kh(u) = 1/h*K(u) or unscaled kernel function K(u). As kernel function the common ones are implemented (gaussian, cauchy, picard, uniform, triangle, cosinus and epanechnikov). Own kernel functions can be built in.

## How to use it

### Create a Kernel Regression dataset
First you have to load your data into the ```KRDataSet``` class. This class prepare and manage the dataset. This class splits, normalizes and reduces the dataset. Also a method to save the data is applied.

The dataset has to consist of matrices and vectors. The rows has to be the single examples and the columns the features. For input and target date are multiple features allowed. 

#### Splitting the data
The dataset have to be split into minimal three subsets. This three subsets are needed for the three calculation steps: 

1. Generating the feature data
2. Validating the regression parameter
3. Testing the learned behavior to estimate the learn success

The data will be split at the initialization of the class. The default setting is ```distribution = (60,20,20)```P. This means that the first subset (features) contains 60%, the second subset (validation) 20% and the third subset (testing) also 20% tuples of the whole dataset. If you set four or more entries into this list, the method produces four or more subsets. The different subsets can get names with the option ```nameString = ('feature', 'validate', 'test')```. A call by the name is current not implemented, but is planned.

```python
data = KrDataSet(inputData = x, targetData = y, distribution = (70, 10, 20), nameString = ('subset1', 'subset2', 'subset3'))
```
#### Reducing the feature subset
Into the ```KrDataSet``` class a method is implemented that provides the option to reduce the feature subset. This is useful to reduce the calculation time. Every row in the feature subset provides one base of the solution. But not every feature is useful like the others. The method ```.reduceFeature(N)``` filters the feature subset. The parameter N defines the degree of reduction. If N >= 1, it represents the number of features that will be reduced. If 0 <= N < 1, it represents the percent of features that will be reduced.

```python
data.reduceFeature(0.5)
```

### Create a Kernel Regression model
The class ```KRModell``` provides the methods to calculate the Kernel Regression model. Some options are implemented for the calculation:
* multiH: boolean; if true each feature gets its own optimized bandwidth, else one bandwidth for all
* scaleKernel: boolean; if true the area below the kernel function is 1, else the output of the kernel function is between 0 and 1
* kernel: string; divines the kernel function
	* gaussian
	* cauchy
	* picard
	* uniform
	* triangle
	* cosinus
	* epanechnikov
		* epanechnikov1
		* epanechnikov2
		* epanechnikov3
* powerList: list; extends the features with each entry as power
	* Example 1: [1] => Kh(u) = f(u_ij)
	* Example 2: [1, 2] => Kh(u) = f(u_ij + (u_ij)^2)
	* Example 3: [0.5, 1, 3] => Kh(u) = f((u_ij)^0.5 + u_ij + (u_ij)^3)

The default is:

```python
options = {'multiH':False, 'scaleKernel':True, 'kernel':'gaussian', 'powerList':[1]}
```

The data has to be the class ```KrDataSet```. The data can be set at initializing:

```python
model = KRModel(krData = data, options = {'multiH':True, 'scaleKernel':False, 'kernel':'gaussian', 'powerList':[1, 2]})
```

or set later:

```python
model.setData(krData = data)
```

Also the options can be change:

```python
model.setOptions(options = {'multiH':True, 'scaleKernel':True, 'kernel':'epanechnikov2', 'powerList':[1/2, 2]})
```
#### Learning
The method ```.learnModel``` runs the learn algorithmus. The method provieds also the option to update the learn options.

```python
model.learnModel(options = {'multiH':True, 'scaleKernel':False, 'kernel':'cauchy', 'powerList':[1]})
```

#### Estimate values
To estimate values from the model some informations are needed. The main information are the points, where the model estimate the function. The point matrix have to be configurate like the input vector.
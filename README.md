### `Dimensionality_Reduction_Techniques`

![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) **`Principal Component Analaysis (PCA)`**

- Goal is to identify and detect strong correlation within variables i.e finding dimensions of maximum variance, reduce the dimensions of a d-dimensional dataset by projecting it onto a (k)-dimensional subspace (where k<d)
- Unlike linear regression it attempts to learn the relationship between X and Y values quantified by finding a list of principal axes.
- PCA can be highly affected by outliers in the data.
- A good analysis of UCIML wine dataset applying PCA with 2 components and building a Logistic Regression classifier.

  - [**PCAclassifier**](https://github.com/kuta-ndze/Dimensionality_Reduction_Techniques/blob/main/PCA/PCA.py)
    | **Visualising the Train set** | **Visualizing the Test set** |
    | :--------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------: |
    | ![**TrainedVisuals**](https://github.com/kuta-ndze/Dimensionality_Reduction_Techniques/blob/main/PCA/trainvisuals.png) | ![**TestVisuals**](https://github.com/kuta-ndze/Dimensionality_Reduction_Techniques/blob/main/PCA/testvisuals.png) |

- In this type of algorithms we could try multiple number of components starting at 2, if we see that the model underperforms to get the optimal number of features.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components = "choose optimal nbr of components")
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
```

![#0f3fff](https://via.placeholder.com/15/0f3fff/000000?text=+) **`Linear Discriminant Analaysis (LDA)`**

- LDA differs from PCA in that in addition to finding the component axises, we are interested in the axes that maximize the separation between multiple classes.
- Both are all linear transformation techniques use for dimensionality reduction.
- PCA is described as unsupervised but LDA is supervised because of the relation to the dependent variable.
- The goal of LDA is to project feature space ( a dataset of n-dimensional samples) onto a small subspace k(where k <= n-1) while maintaining the class-discriminatory information.
- Five steps method for the algorithm as well. The application of LDA before the classifier below.
  - [**LDAclassifier**](https://github.com/kuta-ndze/Dimensionality_Reduction_Techniques/blob/main/LDA/LDA.py)
    | **Visualising the Train set** | **Visualizing the Test set** |
    | :----------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------: |
    | ![**TrainedVisuals**](https://github.com/kuta-ndze/Dimensionality_Reduction_Techniques/blob/main/LDA/trainset.png) | ![**TestVisuals**](https://github.com/kuta-ndze/Dimensionality_Reduction_Techniques/blob/main/LDA/testset.png) |
- The implementation of LDA is different from PCA module.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
#to apply LDA need both features and dependent variables
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
```

### `Dimensionality_Reduction_Techniques`

![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) **`Principal Component Analaysis (PCA)`**

- Goal is to identify and detect strong correlation within variables i.e finding dimensions of maximum variance, reduce the dimensions of a d-dimensional dataset by projecting it onto a (k)-dimensional subspace (where k<d)
- Unlike linear regression it attempts to learn the relationship between X and Y values quantified by finding a list of principal axes.
- PCA can be highly affected by outliers in the data.
- A good analysis of UCIML wine dataset applying PCA with 2 components and building a Logistic Regression classifier.
  - [**Classifier**](https://github.com/kuta-ndze/Dimensionality_Reduction_Techniques/blob/main/PCA/PCA.py)
- |                                             **Visualising the Train set**                                              |                                            **Visualizing the Test set**                                            |
  | :--------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------: |
  | ![**TrainedVisuals**](https://github.com/kuta-ndze/Dimensionality_Reduction_Techniques/blob/main/PCA/trainvisuals.png) | ![**TestVisuals**](https://github.com/kuta-ndze/Dimensionality_Reduction_Techniques/blob/main/PCA/testvisuals.png) |

- In this type of algorithms we could try multiply number of components if we see that the model underperforms to get the optimal number of features. In our example only two features where required.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components = "customize nbr_components")
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
```

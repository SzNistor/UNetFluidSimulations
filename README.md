# Approximating Fluid Simulations with UNets

Over the past few years, the accuracy of computer generated fluid simulations has increased significantly. However, these simulations only approximate the numerical solution of the complex differential equations describing fluid flow to a certain extent. In fact, increasing the approximation's accuracy necessitates a significant amount of computational effort. Recent studies have successfully reduced this effort while maintaining a high level of accuracy by utilizing deep learning models. This study aims to determine the extent at which a simple [UNet](https://arxiv.org/abs/1505.04597) with the mean-squared error loss function can learn to approximate fluid simulations with a single smoke source. The model's capacity to learn from simulations is shown to increase when the source emits a constant volume of smoke over its area and the distance between learned steps is increased from one to ten. As a result, the predicted simulations often exhibit a higher degree of similarity to the target simulations while maintaining a relative error on the frequency power comparable to introducing a medium amount of perturbance. Tests on the power spectrum development of the different predictions were successful in identifying a power-law relationship between frequency and power in log-log space. Finally, the study presents a method for quickly approximating fluid simulations within a known error bound using a simple [UNet](https://arxiv.org/abs/1505.04597).

# Crystal Plasticity

This is a python implementation of the crystal plasticity stress update algorithm using the *interior point method*.
The background of this method and details of the theory and implementation can be found in:

E.S. Perdahcıoğlu, *A rate-independent crystal plasticity algorithm based on the interior point method*, Computer Methods in Applied Mechanics and Engineering, Volume 418, Part A,
2024, 116533, ISSN 0045-7825, [DOI](https://doi.org/10.1016/j.cma.2023.116533).

The intent of publishing this code is to present the implementation in a readable manner and verify the examples shown in the article. It is not intended to be an efficient implementation of the method.

# Usage

The module `CP.py` includes the crystal class implementation which is used to define a grain object with material parameters: *elasticity, hardening* and *euler angles*. 
The current implementation is limited to *fcc* crystals only. 

In order to get the updated stress, the method `update_stress` is called after setting the internal variables of the grain which are the current amount of plastic slip on each slip system and the current accumulated elastic rotation of the material with respect to the crystal orientation.

The module `driver.py` provides a class that is used to generate a sequence of deformation gradients and applying these on a crystal object in order to calculate the updated stresses.

The module `utils.py` provide a collection of functions that are needed for performing simple task associated with the algorithm such as definition of the elasticity tensor etc.

A demonstration of the usage of these is given in the module `test.py` where the results obtained for the example problems demonstrated in the article are regenerated.


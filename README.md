# pybotron

Welcome to pybotron!
Yet another Python robotics library.

But this one is different! It was built around one idea:

#### How to make coding animations in Python as fast as MATLAB

So you can test your algorithms in an isolated environment without the complexity and setup time of things like ROS, while also leveraging the power of Python's libraries and creating a code that can be directly plugged into ROS.

## Features

Custom classes with animation friendly methods, including:

- SimpleRobot: a minimal robot class that allows you to create the wireframe of a revolute joint robot. With convenient functionalities such as calculating the Jacobian and plotting. Includes a UR3e subclass as an example.
- Camera: a minimal class that allows for simulating the projection behavior of a camera to test image-based visual servoing methods.
- PluckerLine: with convenient methods for constructing and transforming.
- Quaternion: with extremely convenient and short operations syntax.
- DualQuaternion: with extremely convenient and short operations syntax, as well as methods for change of form

Mathematical functions (mainly linear algebra) that should have had a simple one-word function in some famous package (but they don't... ) includig:

- Skew symmetric matrix (axiator) of a vector
- Rodrigues formula
- Adjoint transform of a matrix
- Vector to Matrix form of a twist and vice-versa
- Image Jacobian (Interaction matrix)

And much much more!

## Examples

The package includes a handful of demos under the ``/examples`` folder.

I will be working on and off on a more thourough documentation. Feel free to dig into the code since it's really simple.

## Installation
Position yourself in the folder where you want to clone the repo and do:

```bash
git clone https://github.com/higifnr/pybotron.git
```

Then do:
```bash
pip install ./pybotron
```

Enjoy.

(PyPI release is planned ASAP)


## Roadmap

- Lines interaction matrix implementation
- PyPI release
- Better documentation
- ROS1/2 implementation (this would be **EXTREMELY** convenient)
# Subspyces🌶️ - Subspaces in Python!

<p align="center">
  <img src="docs/logo.png" alt="Alt text" width="100">
</p>

## Instalation

`pip install subspyces`

## Why subspyces?

Pytorch is a very powerfull framework for developing ML algorithms, but implementing subspace methods in torch can be a bit repetitive when we don't have a starting point. For this, we developed this simple library which encapsulates some useful code that can be re-used and easily integrated with other torch codebases.


## Overall structure

### Core

The core functionality of this library is the `VectorSpace` class.
This class encapsulates the core of all subspace methods: a simple vector space. Each `VectorSpace` contains a label, a dimension, and set of vectors, which are stored in `torch.Tensor` format.

Idealy, `VectorSpace` should handle all such operations regarding subspaces.

### Generators

A `generator` is a class resposible for _generating_ `VectorSpace` from a pytorch `Dataset`. As such, it will always receive a `torch.utils.data.Dataset` and outputs a list of `VectorSpace`, one vector space per label.

For example, the `generator.IdentityGenerator` class receives any torch dataset and reorganizes it in many vector spaces, without applying any other transformation (Identity).

### Transforms

Different from the `generator` class, a `subspyce.transform` _transforms_ a `VectorSpace` into another `VectorSpace`. For example, the `PCATransform` will receive a `VectorSpace` and apply a PCA decomposition into it.

### Metrics

Finally, the `metrics` module implements some commonly used metric functions.
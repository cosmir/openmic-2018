# openmic-2018
Tools and tutorials for the OpenMIC-2018 dataset.

[![Build Status](https://travis-ci.com/cosmir/openmic-2018.svg?branch=master)](https://travis-ci.com/cosmir/openmic-2018)

[![Coverage Status](https://coveralls.io/repos/github/cosmir/openmic-2018/badge.svg?branch=master)](https://coveralls.io/github/cosmir/openmic-2018?branch=master)

## Overview

This repository contains companion source code for working with the [OpenMIC-2018 dataset](https://zenodo.org/record/1432913#.W6dPeJNKjOR), a collection of audio and crowd-sourced instrument labels produced in a collaboration between [Spotify](https://spotify.com/) and New York Universiy's [MARL](https://steinhardt.nyu.edu/marl/) and [Center for Data Science](https://cds.nyu.edu). The cost of annotation was sponsored by Spotify, whose contributions to open-source research can be found online at the [developer site](http://developer.spotify.com/), [engineering blog](https://labs.spotify.com/), and [public GitHub](https://spotify.github.io/).

If you use this dataset, please cite the following work:

> Humphrey, Eric J., Durand, Simon, and McFee, Brian. "OpenMIC-2018: An Open Dataset for Multiple Instrument Recognition." in Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR), 2018. [pdf](https://zenodo.org/record/1492445#.XsPDCRMzZTY)


## Download the Dataset

The OpenMIC-2018 dataset is made [available on Zenodo](https://zenodo.org/record/1432913#.W6dPeJNKjOR). After downloading, decompress with your favorite commandline tar utility:

```bash
$ tar xvzf openmic-2018-v1.0.0.tgz -C some/dir
```

This will expand into `some/dir/openmic-2018`, with the following structure:

```
openmic-2018/
  acknowledgement.md
  audio/
    000/
      000046_3840.ogg
      ..
    ..
  checksums
  class-map.json
  license-cc-by.txt
  openmic-2018-aggregated-labels.csv
  openmic-2018-individual-responses.csv
  openmic-2018-metadata.csv
  openmic-2018.npz
  partitions/
    train01.txt
    test01.txt
  vggish/
    000/
      000046_3840.json
      ..
    ..
```

The `openmic-2018.npz` is a Python-friendly composite of the `vggish` features and the `openmic-2018-aggregated-labels.csv`. An example of how to train and evaluate a model is provided in a [tutorial notebook](https://github.com/cosmir/openmic-2018/blob/master/examples/modeling-baseline.ipynb).


## Installing

To use the provided `openmic` Python library, first clone the repository and change directory into it:

```bash
$ git clone https://github.com/cosmir/openmic-2018.git
$ cd ./openmic-2018
```

Next, you'll want to pull down the VGGish model parameters via the following script.

```bash
$ ./scripts/download-deps.sh
```

Finally, you can now install the Python library, e.g. with `pip`:

```bash
$ pip install .
```

## Errata

When initially collecting data, ten audio files were corrupted due to [an issue](https://github.com/mdeff/fma/issues/27) in the source FMA dataset:

```python
'071826', '071827', '087435', '095253', '095259',
'095263', '102144', '113025', '113604', '138485'
```

Of the 41k responses obtained, only _three_ resulted in erroneous labels by annotators.
The following rows have been manually corrected:

Sample Key | Instrument | True Label
--- | --- | ---
095253_134400 | piano | yes
095263_96000 | mallet percussion | yes
113025_99840 | trumpet | yes

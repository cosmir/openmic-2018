# openmic-2018
Tools and tutorials for the OpenMIC-2018 dataset.

[![Build Status](https://travis-ci.com/cosmir/openmic-2018.svg?branch=master)](https://travis-ci.com/cosmir/openmic-2018)

[![Coverage Status](https://coveralls.io/repos/github/cosmir/openmic-2018/badge.svg?branch=master)](https://coveralls.io/github/cosmir/openmic-2018?branch=master)


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

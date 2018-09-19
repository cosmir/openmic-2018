from setuptools import setup, find_packages

import imp

repo_name = 'openmic-2018'
package_name = 'openmic'
description = 'OpenMIC-2018: Software Tools and Tutorials'
version = imp.load_source('{}.version'.format(package_name),
                          '{}/version.py'.format(package_name))

setup(
    name=package_name,
    version=version.version,
    description=description,
    author='COSMIR',
    url='http://github.com/cosmir/{}'.format(repo_name),
    download_url='http://github.com/cosmir/{}/releases'.format(repo_name),
    packages=find_packages(),
    package_data={'': ['vggish/_model/vggish_model.ckpt',
                       'vggish/_model/vggish_pca_params.npz']},
    long_description=description,
    classifiers=[
        "License :: OSI Approved :: ?",
        "Programming Language :: Python",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    keywords='openmic audio dataset music vggish',
    license='MIT',
    install_requires=[
        'pandas>=0.22',
        'numpy>=1.15',
        'scipy>=1.1',
        'scikit-learn>=0.19.1',
        'tensorflow==1.9',
        'tqdm',
        'resampy',
        'pysoundfile>=0.9',
        'joblib'
    ],
    extras_require={},
    scripts=['scripts/featurefy.py']
)

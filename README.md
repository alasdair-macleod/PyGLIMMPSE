# pyglimmpse
Function library containing tools for calculating power and samplesize for general linear mixed models.

# build

`
python setup.py sdist

python setup.py bdist_wheel

python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*29*
`
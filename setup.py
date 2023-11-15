import setuptools

setuptools.setup(
    name='pyglimmpse',
    version="0.0.33",
    packages=setuptools.find_packages(exclude=['tests*']),
    include_package_data=True,
    install_requires=['scipy', 'numpy'],
)
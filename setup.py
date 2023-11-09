from setuptools import setup

setup(name='dino_paint',
    version='0.1',
    description='DINOv2 with VGG16 for semantic segmentation using a random forest classifier',
    author='Roman Schwob',
    author_email='roman.schwob@students.unibe.ch',
    license='GNU GPLv3',
    packages=['package'],
    scripts=['bin/test.py'],
    zip_safe=False)
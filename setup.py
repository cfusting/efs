from setuptools import setup, find_packages

setup(
    name='efs',
    version='0.1.0',
    description='Evolutionary Feature Synthesis.',
    author='Chris Fusting',
    author_email='cfusting@gmail.com',
    license='GNU GPLv3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3'
    ],
    keywords='evolution machine learning artificial intelligence',
    install_requires=[
        'numpy',
        'scikit_learn',
        'scipy'
    ],
    python_requires='>=2.7',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    url='https://github.com/cfusting/efs'
)

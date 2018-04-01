from distutils.core import setup
setup(
    name='to',
    packages=['to'],  # this must be the same as the name above
    package_dir={'to': 'to'},
    package_data={'to': ['utils/*.py', 'data/*.py', 'net/*.py', 'layers/*.py']},
    version='0.3',
    description='A trainer for PyTorch',
    author='Jacob Zhang',
    author_email='r.c.cham@gmail.com',
    url='https://github.com/chamrc/to',  # use the URL to the github repo
    download_url='https://github.com/chamrc/to/archive/0.3.tar.gz',  # I'll explain this in a second
    keywords=['testing', 'logging', 'example'],  # arbitrary keywords
    classifiers=[],
)

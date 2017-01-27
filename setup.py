from setuptools import setup


PACKAGES = [
    'vecgram'
]

def setup_package():
    setup(
        name="VecGram",
        version='0.1.0',
        description='NLP Models with a Focus on Deep Learning',
        author='Joshua D. Loyal',
        url='https://github.com/joshloyal/VecGram',
        license='MIT',
        install_requires=['spacy'],
        packages=PACKAGES,
    )


if __name__ == '__main__':
    setup_package()

from setuptools import setup

# 这是 Python 项目的标准打包文件
setup(name='minGPT',
      version='0.0.1',
      author='Andrej Karpathy',
      packages=['mingpt'], # 指定要打包的目录，这里是 mingpt 文件夹
      description='A PyTorch re-implementation of GPT',
      license='MIT',
      install_requires=[ # 依赖项：这个项目需要 pytorch
            'torch',
      ],
)

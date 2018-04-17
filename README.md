Installation instructions

1. Download and install the Python 3.6 version of Anaconda.

https://www.anaconda.com/download/

2. Create the Anaconda environment and activate it.

```
$ conda env create -f environment.yml

$ source activate collab_design
```

3. Download the trained lamem model.

http://memorability.csail.mit.edu/memnet.tar.gz

4. Unzip and untar the model in the current directory.

```
$ gunzip memnet.tar.gz

$ tar xvf memnet.tar
```

You should have a directory called memnet along side this file.

5. Launch ipython and run demo code

```
$ ipython

In[1]: %run prototype.py

In[2]: demo('hats')
```

Look in the data directory for the results.

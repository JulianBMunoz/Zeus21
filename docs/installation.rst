============
Installation
============

You can download and install this package by doing:

.. code-block:: console

    git clone https://github.com/julianbmunoz/zeus21.git zeus21
    cd zeus21/
    pip install .

that should take care of all dependencies (remember to work in your favorite conda env). 
If you have issues with cache'd versions of packages you can add ``--no-cache-dir`` at the end of ```pip install .``. 

**NOTE:** You may run into problems when pip-installing ``classy`` (the Python wrapper of ``CLASS``). 
If so, their installation guide is `here <https://github.com/lesgourg/class_public/wiki/Installation>`_, 
but in short the steps are:

.. code-block:: console

    git clone https://github.com/lesgourg/class_public.git class
    cd class/
    make
    cd python/
    python setup.py install --user

(modifying the Makefile to your `gcc` as needed)
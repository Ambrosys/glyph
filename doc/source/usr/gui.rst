
Glyph remote - GUI
==================


Install
'''''''''''''''''''''''''

Glyph comes with an optional GUI to use the ``glyph-remote`` script with mor convenience.

The GUI uses the package ``wxPython``, in order to install glyph all the prerequisites of wxPython must be installed.

**For Ubuntu those are:**

- dpkg-dev
- build-essential
- python3.5-dev # minimum version to use glyph
- libjpeg-dev
- libtiff-dev
- libsdl1.2-dev
- libgstreamer-plugins-base0.10-dev
- libnotify-dev
- freeglut3
- freeglut3-dev
- libsm-dev
- libgtk-3-dev
- libwebkitgtk-3.0-dev # or libwebkit2gtk-4.0-dev if available
- libxtst-dev

For further questions see the developers github `README.md <https://github.com/wxWidgets/Phoenix/blob/master/README.rst#prerequisites>`_
and `Website <https://wxpython.org/>`_.

**Note:**
If you consider to use a conda envrionment:
``conda install wxpython`` should do the trick

Manual Gooey installtion
^^^^^^^^^^^^^^^^^^^^^^^^

Since up-to-date (28.08.2018) the neccesarry changes to the used graphic library Gooey are not part of the master branch,
it might be neccessary to install Gooey by hand from the repo `https://github.com/Magnati/Gooey <https://github.com/Magnati/Gooey>`_ in three steps.

- ``git clone git@github.com:Magnati/Gooey.git``
- ``cd Gooey``
- ``python setup.py install``

Installation with pip installtion
^^^^^^^^^^^^^^^^^^^^^^^^

To install glyph including the gui option use the following command:

.. code-block::
    python pip install pyglyph[gui]``

To start the script with the gui just use the ``--gui`` parameter:

.. code-block::
    glyph-remote --gui

Usage
''''''

Within the GUI there are the following tabs:

- Optional Arguments,
- GP config
- algorithm
- breeding
- assessment
- break
- condition
- constraints

If all primitves are set, click the Start-button to start the remoteserver.
If you start another shell and cd to
``/glyph/examples/remote/``
you may start the example experiment ``python experiment.py``.

In the **GP config** tab you may specify a `config.yaml` to set All parameters from the tabs

- algorithm
- breeding
- assessment
- break condition
- constraints

As well as the used primitives. An example can be found in

``/glyph/examples/remote/experiment.yaml``

For your own experiemnts the `experiment_dummy.py` is a good file to start.





Glyph remote - GUI
==================


Install
'''''''

Glyph comes with an optional GUI to use the ``glyph-remote`` script with more convenience.

The GUI uses the package ``wxPython``. The installation manual can be found `here <https://github.com/wxWidgets/Phoenix/blob/master/README.rst#prerequisites>`_
and `Website <https://wxpython.org/>`_.


Manual Gooey installtion
^^^^^^^^^^^^^^^^^^^^^^^^

Since up-to-date (28.08.2018) the necessary changes to the used graphic library Gooey are not part of the master branch,
it might be necessary to install Gooey by hand from the repo `https://github.com/Magnati/Gooey <https://github.com/Magnati/Gooey>`_ in three steps.

- ``pip install -e "git+git@github.com:Magnati/Gooey.git#egg=gooey"``


Installation with pip installtion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install glyph including the gui option use the following command:

.. code-block::
    python pip install pyglyph[gui]``

To start the script with the gui just use the ``--gui`` parameter:

.. code-block::
    glyph-remote --gui

Usage
''''''

Within the GUI there is a tab for each group of parameters.
If all parameters are set, click the start-button to start the experiment.






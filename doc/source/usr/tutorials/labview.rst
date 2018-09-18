Labview Tutorial
----------------

Contributed by P. Oswald.

Install Python
==============

1. Install `Miniconda (Python 3.5 or higher) <https://conda.io/miniconda.html>`_.
2. Open a command window as administrator:
  * cd to Miniconda3 directory
  * run :code:`conda update conda`
3. Install the numpy and scipy wheels using conda, or download them directly `here <http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy/>`_ and `here <http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy/>`_. You can install them with :code:`pip install path_to_wheel/wheel_file_name`.

Install Glyph
=============

0.  If you have git installed, run :code:`pip install –e git+https://github.com/Ambrosys/glyph.git#egg=glyph`. Go to step 5.
1.	Download the latest version from `Github <https://github.com/Ambrosys/glyph>`_.
2.	Unzip / move to somewhere useful
3.	Upen a cmd window, navigate the the glyph-master folder
4.	Run :code:`pip install –e .` (don’t forget the period at the end)
5.	Test the installation by running :code:`glyph-remote --help`.


Install ZeroMQ
==============

1. Download ZeroMQ bindings for LabView from http://labview-zmq.sourceforge.net/
2. The download is a VI-Package (*.vip-file)
3. Double clicking the *.vip-file opens it in the VI Package Manager (further info http://www.ni.com/tutorial/12397/en/)
4. Use the VI Package Manager to install the package


Use ZeroMQ
==========

1. After successful installation you can find examples on the usage of ZeroMQ either
  a. through the VI Package Manager by double clicking on the entry "ZeroMQ Socket Library" and then on the button "Show Examples"
  b. in your LabView installation folder in the subdirectory /examples/zeromq/examples/
  c. online (e.g. the basic examples at http://labview-zmq.sourceforge.net/)
2. For communication with glyph-remote one has to implement a server that listens for requests from glyph and sends the apropriate responses
3. The ZeroMQ programming blocks can be accessed by right clicking on the block diagram and navigating to the section "Add-ons"
4. The block "Unflatten from JSON" can be used to convert the JSON encoded strings sent by glyph to LabView clusters

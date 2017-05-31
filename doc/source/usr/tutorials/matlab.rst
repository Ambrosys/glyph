Matlab Tutorial
---------------

Contributed by B. Strom.

.. note::
    This was performed on Windows 7 and using MATLAB R2016b (2016b or later is needed for :code:`jasondecode()` and :code:`jasonencode()` commands)

Install Python
==============

1. Install `Miniconda (Python 3.5 or higher) <https://conda.io/miniconda.html>`_.
2. Open a command window as administrator:
  * cd to Miniconda3 directory
  * run :code:`conda update conda`
3. Install the numpy and scipy wheels using conda, or download them directly `here <http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy/>`_ and `here <http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy/>`_. You can install them with :code:`pip install path_to_wheel/wheel_file_name`

Install Glyph
=============

0.  If you have git installed, run :code:`pip install –e git+https://github.com/Ambrosys/glyph.git#egg=glyph`. Go to step 5.
1.	Download the latest version from `Github <https://github.com/Ambrosys/glyph>`_.
2.	Unzip / move to somewhere useful
3.	Upen a cmd window, navigate the the glyph-master folder
4.	Run :code:`pip install –e .` (don’t forget the period at the end)
5.	Test the installation by running :code:`glyph-remote --help`

Install jeroMQ (java implementation of zeroMQ)
==============================================

This will be used for zeroMQ in MATLAB.

1.	If you don’t have it, install the `Java developer kit <http://www.oracle.com/technetwork/java/javase/downloads/index.html>`_.
2.	Set the JAVA_HOME environment variable
  a.	Right click My Computer and select properties
  b.	On the Advanced tab, select Environment Variables, and then edit or create the system variable JAVA_HOME  to point to where the JDK software is located, for example, :code:`C:\Program Files\Java\jdk1.8.0_131`

3.	Install `Maven <https://maven.apache.org/>`_.
  a.	Add the bin directory of the created directory apache-maven-3.5.0 to the PATH environment variable (same steps as the setting the :code:`JAVA_HOME` variable, but this is a user variable instead of a system variable)
  b.	Confirm installation with :code:`mvn -v` in a command window

4.	Download `jeroMQ <https://github.com/zeromq/jeromq>`_.
  a.	Unpack the zip file
  b.	In a command window, navigate to the resulting jeroMQ folder
  c.	Run the command :code:`mvn package`
  d.	This will take a while, but you should see “Build Success” when it is finished
  e.	This will have created a “target” directory in the jeroMQ folder. The Jar file we need  is in here, something like :code:`…/target/jeromq-0.4.1-SNAPSHOT.jar`

5.	Add the path to this Jar file to MATLAB's static Java path
  a.	Run the command :code:`prefdir` in MATLAB. Navigate to that folder and check for a file named :code:`javaclasspath.txt`.
  b.	Open this file in a text editor or create anASCII text file named :code:`javaclasspath.txt`.
  c.	On its own line, add the full path to the jar file, including the file name. You can move it or rename it first if you wish.
  d.	Restart MATLAB

6.	To test that MATLAB can access jeroMQ, run :code:`import org.zeromq.ZMQ` in at the MATLAB command prompt.  If no error, it was successful.

Test a basic example
====================

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
import glyph

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.pngmath',
]
templates_path = ['_templates']
autodoc_default_flags = ['members', 'show-inheritance', 'special-members', 'any']
autodoc_member_order = 'bysource'

source_suffix = ['.rst']

master_doc = 'index'
project = 'glyph'
import datetime
copyright = '{}, Markus Abel, Julien Gout, Markus Quade, <a href="http://www.ambrosys.de">Ambrosys GmbH</a>'.format(datetime.datetime.now().year)
author = 'Markus Abel, Julien Gout, Markus Quade'

version = glyph.__VERSION__
release = glyph.__VERSION__

language = None
exclude_patterns = ['_build']

pygments_style = 'sphinx'
add_module_names = True
add_function_parentheses = False
todo_include_todos = True

html_theme = 'alabaster'
html_theme_options = {
    'logo': 'tmp_logo.jpg',
    'logo_name': True,
    'show_powered_by': False,
    'github_user': 'Ambrosys',
    'github_repo': 'glyph',
    'github_banner': True,
    'github_type': 'star',
    'show_related': False,
    'description': "Symbolic regression tools.",
}


html_sidebars = {
    'index': [
        'about.html',
        'sidebarintro.html',
        'navigation.html',
        'searchbox.html',
        'hacks.html',  # kudos to kenneth reitz
    ],
    '**': [
        'about.html',
        #'navigation.html',
        'localtoc.html',
        'searchbox.html',
        'hacks.html',
    ]
}
html_static_path = ['_static']

htmlhelp_basename = 'glyphdoc'

html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = True

intersphinx_mapping = {
    'python': ('http://docs.python.org/3.5', None),
    'deap': ('http://deap.readthedocs.io/en/master/', None),
    'numpy'  : ('http://docs.scipy.org/doc/numpy', None)
}
default_role='any'

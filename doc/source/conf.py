import datetime
import os
import sys


import sphinx_readable_theme

sys.path.insert(0, os.path.abspath("../"))
from glyph._version import get_versions

project = "glyph"
master_doc = "index"
copyright = '{}, <a href="http://www.ambrosys.de">Ambrosys GmbH</a>'.format(
    datetime.datetime.now().year
)
author = "Ambrosys GmbH"
version = release = get_versions()["version"]

extensions = [
    "sphinxcontrib.apidoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.imgmath",
    "sphinx.ext.napoleon",
]

apidoc_module_dir = "../../glyph"
apidoc_output_dir = "dev/api/"
apidoc_excluded_paths = ["tests", "setup.py"]
apidoc_separate_modules = False

autodoc_default_flags = ["members"]
autoclass_content = "init"
napoleon_include_special_with_doc = True
source_suffix = [".rst"]

language = None
exclude_patterns = ["_build"]

pygments_style = "trac"
add_module_names = False
add_function_parentheses = False
todo_include_todos = True


html_theme = "readable"
html_theme_path = [sphinx_readable_theme.get_html_theme_path()]
# html_theme = "alabaster"
# html_theme_options = {
#     "logo": "tmp_logo.jpg",
#     "logo_name": True,
#     "show_powered_by": False,
#     "github_user": "Ambrosys",
#     "github_repo": "glyph",
#     "github_banner": True,
#     "github_type": "star",
#     "show_related": False,
#     "description": "Symbolic regression tools.",
# }


# for sidebarintro.html

html_additional_pages = {
    "sidebarintro": "sidebarintro.html"
}

html_context = {
    "github_user": "ambrosys",
    "github_repo": project.lower(),
    "github_button": True,
    "github_banner": True,
    "github_type": "star",
    "github_count": True,
    "badge_branch": "master",
    "pypi_project": "pyglyph",
}
templates_path = ["_templates"]
html_sidebars = {
    "index": ["sidebarintro.html"],
    "**": ["sidebarintro.html", "localtoc.html", "searchbox.html"],
}


html_static_path = ["_static"]

htmlhelp_basename = "glyphdoc"

html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = True

intersphinx_mapping = {
    "python": ("http://docs.python.org/3.6", None),
    "deap": ("http://deap.readthedocs.io/en/master/", None),
    "np": ("http://docs.scipy.org/doc/numpy", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference", None),
}
default_role = "any"

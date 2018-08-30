import logging

logger = logging.getLogger(__name__)

try:
    from gooey import Gooey, GooeyParser
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    def Gooey(**kwargs):
        def wrapper(func):
            pass
        logger.error("Gooey library is not installed")
        return wrapper

    def GooeyParser(**kwargs):
        return None





def is_gooey_active():
    global GUI_AVAILABLE
    return GUI_AVAILABLE


@Gooey(
    auto_start=False,  # Skips the configuration all together and runs the program immediately
    advanced=True,  # toggle whether to show advanced config or not
    encoding="utf-8",  # Text encoding to use when displaying characters (default: 'utf-8')
    language="english",  # Translations configurable via json
    show_config=True,  # skip config screens all together
    # target=executable_cmd,     # Explicitly set the subprocess executable arguments
    # program_name='name',       # Defaults to script name
    # program_description="",       # Defaults to ArgParse Description
    default_size=(1200, 1000),  # starting size of the GUI
    # required_cols=1,           # number of columns in the "Required" section NOTE:Deprecation notice: See Group Parameters for modern layout controls
    # optional_cols=2,           # number of columbs in the "Optional" section NOTE:Deprecation notice: See Group Parameters for modern layout controls
    dump_build_config=False,  # Dump the JSON Gooey uses to configure itself
    load_build_config=None,  # Loads a JSON Gooey-generated configuration
    monospace_display=False,  # Uses a mono-spaced font in the output screen
    # image_dir=, 	       # Path to the directory in which Gooey should look for custom images/icons
    # language_dir=,  	   # Path to the directory in which Gooey should look for custom languages files
    disable_stop_button=False,  # Disable the Stop button when running
    show_stop_warning=True,  # Displays a warning modal before allowing the user to force termination of your program
    force_stop_is_error=True,  # Toggles whether an early termination by the shows the success or error screen
    show_success_modal=True,  # Toggles whether or not to show a summary modal after a successful run
    run_validators=True,  # Controls whether or not to have Gooey perform validation before calling your program
    poll_external_updates=False,  # (Experimental!) When True, Gooey will call your code with a gooey-seed-ui CLI argument and use the response to fill out dynamic values in the UI (See: Using Dynamic Values)
    return_to_config=False,  # When True, Gooey will return to the configuration settings window upon successful run
    # progress_regex=, 	   # A text regex used to pattern match runtime progress information. See: Showing Progress for a detailed how-to
    # progress_expr=, 	   # A python expression applied to any matches found via the progress_regex. See: Showing Progress for a detailed how-to
    disable_progress_bar_animation=False,  # Disable the progress bar
    navigation="SIDEBAR",  # Sets the "navigation" style of Gooey's top level window. Options: TABBED	SIDEBAR
    tabbed_groups=True,
    navigation_title="Actions",  # Controls the heading title above the SideBar's navigation pane. Defaults to: "Actions"
    show_sidebar=False,  # Show/Hide the sidebar in when navigation mode == SIDEBAR
    # body_bg_color, 	        # HEX value of the main Gooey window
    # header_bg_color, 	        # HEX value of the header background
    # header_height,             # height in pixels of the header
    # header_show_title, 	    # Show/Hide the header title
    # header_show_subtitle,      # Show/Hide the header subtitle
    # footer_bg_color, 	        # HEX value of the Footer background
    # sidebar_bg_color, 	        # HEX value of the Sidebar's background
    # terminal_panel_color, 	    # HEX value of the terminal's panel
    # terminal_font_color, 	    # HEX value of the font displayed in Gooey's terminal
    # terminal_font_family, 	    # Name of the Font Family to use in the terminal
    # terminal_font_weight,	    # Weight of the font (NORMAL
    # terminal_font_size, 	    # Point size of the font displayed in the terminal
    # error_color, 	            # HEX value of the text displayed when a validation error occurs
)
def get_gooey(RemoteApp):
    parser = GooeyParser(prog="glyph-remote-gui")
    return parser


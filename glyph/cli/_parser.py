import argparse
import logging

from glyph.utils.argparse import catch_and_log, positive_int, non_negative_int, readable_file

logger = logging.getLogger(__name__)

try:
    import gooey
    from gooey import Gooey, GooeyParser

    @Gooey(
        auto_start=False,
        advanced=True,
        encoding="utf-8",
        language="english",
        show_config=True,
        default_size=(1200, 1000),
        dump_build_config=False,
        load_build_config=None,
        monospace_display=False,
        disable_stop_button=False,
        show_stop_warning=True,
        force_stop_is_error=True,
        show_success_modal=True,
        run_validators=True,
        poll_external_updates=False,
        return_to_config=False,
        disable_progress_bar_animation=False,
        navigation="SIDEBAR",
        tabbed_groups=True,
        navigation_title="Actions",
        show_sidebar=False,
    )
    def get_gooey(prog="glyph-remote"):
        problably_fork = "site-packages" not in gooey.__file__
        logger.debug("Gooey located at {}.".format(gooey.__file__))
        if problably_fork:
            logger.warning("GUI input validators may have no effect")
        parser = GooeyParser(prog=prog)
        return parser

    GUI_AVAILABLE = True
except ImportError as e:
    logger.error(e)
    GUI_AVAILABLE = False
    GUI_UNAVAILABLE_MSG = """Could not start gui extention.
You need to install the gui extras.
Use the command 'pip install glyph[gui]' to do so."""


class GooeyOptionsArg:
    POSITIVE_INT = {
        "validator": {
            "callback": catch_and_log(positive_int),
            "message": "This is not a positive integer.",
        }
    }
    NON_NEGATIVE_INT = {
        "validator": {
            "callback": catch_and_log(non_negative_int),
            "message": "This is not a non negative integer.",
        }
    }
    READABLE_FILE = {
        "validator": {
            "callback": catch_and_log(readable_file),
            "message": "This is not a readable file.",
        }
    }


class MyGooeyMixin:
    def add_argument(self, *args, **kwargs):
        for key in ["widget", "gooey_options"]:
            if key in kwargs:
                del kwargs[key]
        super().add_argument(*args, **kwargs)

    def add_mutually_exclusive_group(self, *args, **kwargs):
        group = MutuallyExclusiveGroup(self, *args, **kwargs)
        self._mutually_exclusive_groups.append(group)
        return group

    def add_argument_group(self, *args, **kwargs):
        group = ArgumentGroup(self, *args, **kwargs)
        self._action_groups.append(group)
        return group


class Parser(MyGooeyMixin, argparse.ArgumentParser):
    pass


class ArgumentGroup(MyGooeyMixin, argparse._ArgumentGroup):
    pass


class MutuallyExclusiveGroup(MyGooeyMixin, argparse._MutuallyExclusiveGroup):
    pass




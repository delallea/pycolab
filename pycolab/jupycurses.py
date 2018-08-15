"""
Minimal implementation of the `curses` API to work within a Jupyter notebook.
"""

import numpy as np
import PIL.ImageFont


# "Official" `curses` constants.

# noinspection PyUnresolvedReferences
from curses import COLOR_BLACK, COLOR_WHITE, KEY_NPAGE, KEY_PPAGE

COLOR_PAIRS = 65536
COLORS = 256

# Custom constants.

_A_COLOR_OFFSET = 24  # in the attributes bitmask, colors start at this offset
_CELL_PIXEL_HEIGHT = 10  # height of a single cell, in pixels
_CELL_PIXEL_WIDTH = 10  # width of a single cell, in pixels
_COLOR_PAIRS = {
    0: (COLOR_WHITE, COLOR_BLACK),
}
_MAX_COLOR_INTENSITY = 1000
_COLORS = {
    COLOR_BLACK: (0, 0, 0),
    COLOR_WHITE: (_MAX_COLOR_INTENSITY, _MAX_COLOR_INTENSITY, _MAX_COLOR_INTENSITY),
}
_SCREEN_HEIGHT = 100  # number of rows
_SCREEN_WIDTH = 80  # number of columns

# More "official" `curses` constants defined from custom ones.

# This builds a 64-bit bitmask of the form: 0xffffffffff000000
# The number of zero bits at the end is equal to `_A_COLOR_OFFSET`.
A_COLOR = int(np.uint64(-1)) - (2 ** _A_COLOR_OFFSET - 1)


class JupyWindow(object):

    def __init__(self, max_x=_SCREEN_WIDTH, max_y=_SCREEN_HEIGHT, begin_x=0, begin_y=0):
        """
        Constructor.

        :param max_x: Number of columns.
        :param max_y: Number of rows.
        :param begin_x: X coordinate of the top-left corner of the window.
        :param begin_y: Y coordinate of the top-left corner of the window.
        """
        self._max_x = max_x
        self._max_y = max_y
        self._begin_x = begin_x
        self._begin_y = begin_y
        # Set blocking or non-blocking behavior:
        #   - if `None`: blocking indefinitely
        #   - otherwise, block only for this amount of seconds (can be zero for non-blocking) before returning
        self._block_timeout = None
        self._data = np.zeros((self._max_x, self._max_y), dtype=np.uint32)
        # The list of items that may be plotted into a grid cell.
        self._items = [None]
        # Map a tuple identifying an item that can be plotted in a grid cell to its index in `_items`.
        self._item_key_to_idx = {None: 0}
        # The font to use to plot characters.
        self._font = PIL.ImageFont.load_default()

    def _add_item(self, y, x, key):
        """
        Add an item identified by `key` at position `(y, x)`.
        """
        try:
            item_idx = self._item_key_to_idx[key]
        except KeyError:
            # Must create item first.
            item_idx = self._create_item(key=key)
        self._data[x, y] = item_idx

    def _create_char_item(self, ch, attr):
        """
        Create a new character item.

        :param ch: The character we want to create.
        :param attr: Attribute controlling the character format.
        :return: The corresponding item.
        """
        # Currently only color attributes are supported.
        if attr & ~A_COLOR:
            raise NotImplementedError(
                f'Currently only color attributes are supported, but {hex(attr)} has non color attributes')
        im = self._font.getmask(ch)
        im = im.resize((_CELL_PIXEL_WIDTH, _CELL_PIXEL_HEIGHT))
        # Create numpy array with (binary) pixel values.
        data = np.zeros(list(reversed(im.size)), dtype=np.uint8)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j] = int(im.getpixel((j, i)) > 0)
        # Convert to RGB.
        color_number = (attr & A_COLOR) >> _A_COLOR_OFFSET
        rgb = np.array(color_content(color_number), dtype=np.uint8)  # TODO CONTINUE HERE MUST DIVIDE BY MAX * 255
        item = data * rgb

    def _create_item(self, key):
        """
        Create a new item associated to `key`.

        :return: The index of the item that is created.
        """
        item_type = key[0]
        params = key[1:]
        if item_type == 'char':
            item = self._create_char_item(*params)
        else:
            raise NotImplementedError(item_type)
        item_idx = len(self._items)
        self._items.append(item)
        self._item_key_to_idx[key] = item_idx
        return item_idx

    def addch(self, y, x, ch, attr=None):
        """
        See `Window.addch()`.
        """
        self._add_item(y=y, x=x, key=('char', ch, attr))

    def addstr(self, y, x, msg, attr=None):
        """
        See `Window.addstr()`.
        """
        for i, ch in enumerate(msg):
            self.addch(y=y, x=x + i, ch=ch, attr=attr)


    def erase(self):
        """
        See `Window.erase()`.
        """
        self._data.fill(0)

    def getmaxyx(self):
        """
        See `Window.getmaxyx()`.
        :return:
        """
        return self._max_y, self._max_x

    def timeout(self, delay):
        """
        See `Window.timeout()`.
        """
        if delay < 0:
            self._block_timeout = None
        else:
            self._block_timeout = delay / 1000.


def can_change_color():
    """
    See `curses.can_change_color()`.
    """
    return True


def color_content(color_number):
    """
    See `curses.color_content().`
    """
    return _COLORS[color_number]


def color_pair(color_number):
    """
    See `curses.color_pair()`.
    """
    return color_number << _A_COLOR_OFFSET


# noinspection PyUnusedLocal
def curs_set(visibility):
    """
    See `curse.curs_set()`.

    Current implementation does nothing because showing the cursor is not supported.
    """
    pass


def init_color(color_number, r, g, b):
    """
    See `curses.init_color()`.
    """
    _COLORS[color_number] = (r, g, b)


def init_pair(pair_number, fg, bg):
    """
    See `curses.init_pair()`.
    """
    _COLOR_PAIRS[pair_number] = (fg, bg)


def newwin(nlines=_SCREEN_HEIGHT, ncols=_SCREEN_WIDTH, begin_y=0, begin_x=0):
    return JupyWindow(max_x=ncols, max_y=nlines, begin_x=begin_x, begin_y=begin_y)


def pair_content(pair_number):
    """
    See `curses.pair_content()`.
    """
    return _COLOR_PAIRS[pair_number]


def wrapper(func):
    """
    See `curses.wrapper()`.
    """
    window = JupyWindow()
    func(window)

"""
Minimal implementation of the `curses` API to work within a Jupyter notebook.
"""

import io
import sys
import time
from collections import deque

import ipywidgets
import numpy as np
import PIL.Image
import PIL.ImageFont


# "Official" `curses` constants.

# noinspection PyUnresolvedReferences
from curses import COLOR_BLACK, COLOR_WHITE, KEY_DOWN, KEY_LEFT, KEY_NPAGE, KEY_PPAGE, KEY_RIGHT, KEY_UP

COLOR_PAIRS = 65536
COLORS = 256

# Custom constants.

_A_COLOR_OFFSET = 24  # in the attributes bitmask, colors start at this offset
_CELL_PIXEL_HEIGHT = 11  # height of a single cell, in pixels
_CELL_PIXEL_WIDTH = 10  # width of a single cell, in pixels
_COLOR_PAIRS = {
    0: (COLOR_WHITE, COLOR_BLACK),
}
_MAX_COLOR_INTENSITY = 1000
_MAX_RGB_VALUE = 255
_COLORS = {
    COLOR_BLACK: (0, 0, 0),
    COLOR_WHITE: (_MAX_COLOR_INTENSITY, _MAX_COLOR_INTENSITY, _MAX_COLOR_INTENSITY),
}
_COLOR_TO_RGB = {}  # map a color number to a numpy array of RGB values
_SCREEN_HEIGHT = 40  # number of rows
_SCREEN_WIDTH = 80  # number of columns
_DEFAULT_SLEEP_PERIOD = 0.1  # wait for this number of seconds while waiting for a command
_ALL_WINDOWS = []  # list of all active windows
_PIXELS = np.zeros((_SCREEN_HEIGHT * _CELL_PIXEL_HEIGHT, _SCREEN_WIDTH * _CELL_PIXEL_WIDTH, 3), dtype=np.uint8)  # RGB
_CALLBACKS = []  # callback functions to be called when the screen is updated
_COMMANDS = deque()  # commands sent by the user
_IMAGE_FORMAT = 'bmp'  # image format when converting bytes to image
_IMAGE = None

# More "official" `curses` constants defined from custom ones.

# This builds a 64-bit bitmask of the form: 0xffffffffff000000
# The number of zero bits at the end is equal to `_A_COLOR_OFFSET`.
A_COLOR = int(np.uint64(-1)) - (2 ** _A_COLOR_OFFSET - 1)


class JupyWindow(object):

    def __init__(self, n_lines=_SCREEN_HEIGHT, n_columns=_SCREEN_WIDTH, begin_y=0, begin_x=0):
        """
        Constructor.

        :param n_lines: Number of lines.
        :param n_columns: Number of columns.
        :param begin_y: Y coordinate of the top-left corner of the window (0 = top).
        :param begin_x: X coordinate of the top-left corner of the window (0 = left).
        """
        self._n_lines = n_lines
        self._n_columns = n_columns
        self._begin_y = begin_y
        self._begin_x = begin_x
        # Set blocking or non-blocking behavior:
        #   - if `None`: blocking indefinitely
        #   - otherwise, block only for this amount of seconds (can be zero for non-blocking) before returning
        self._block_timeout = None
        self._data = np.zeros((self._n_lines, self._n_columns), dtype=np.uint32)
        # The list of items that may be plotted into a grid cell.
        self._items = [None]
        # Map a tuple identifying an item that can be plotted in a grid cell to its index in `_items`.
        self._item_key_to_idx = {None: 0}
        # The font to use to plot characters.
        self._font = PIL.ImageFont.load_default()
        # Current position of the cursor.
        self._current_y = 0
        self._current_x = 0
        # The window buffer (RGB values).
        self._pixels = np.zeros((self._n_lines * _CELL_PIXEL_HEIGHT, self._n_columns * _CELL_PIXEL_WIDTH, 3),
                                dtype=np.uint8)

    def _addch(self, y, x, ch, attr=None):
        """
        Add a character in the specific position.

        `ch` may either be an integer or a character.
        """
        if isinstance(ch, int):
            ch = chr(ch)
        self._add_item(y=y, x=x, key=('char', ch, attr))

    def _add_item(self, y, x, key):
        """
        Add an item identified by `key` at position `(y, x)`.
        """
        try:
            item_idx = self._item_key_to_idx[key]
        except KeyError:
            # Must create item first.
            item_idx = self._create_item(key=key)
        self._data[y, x] = item_idx

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
        data = np.zeros((im.size[1], im.size[0], 1), dtype=np.uint8)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j, 0] = int(im.getpixel((j, i)) > 0)
        # Convert to RGB.
        pair_number = (attr & A_COLOR) >> _A_COLOR_OFFSET
        fg, bg = pair_content(pair_number)
        fg_rgb, bg_rgb = (_get_rgb_array(c) for c in [fg, bg])
        item = np.where(data > 0, fg_rgb, bg_rgb)
        # assert ch == ' ' or item.sum() > 0, (ch, attr)
        return item

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
        # noinspection PyTypeChecker
        self._items.append(item)
        self._item_key_to_idx[key] = item_idx
        return item_idx

    def addch(self, *args):
        """
        See `Window.addch()`.
        """
        if len(args) <= 2:
            ch = args[0]
            y = self._current_y
            x = self._current_x
        else:
            y, x, ch = args[0:3]
        assert isinstance(ch, int)
        attr = args[-1] if len(args) % 2 == 0 else None
        self._addch(y=y, x=x, ch=ch, attr=attr)
        self._current_y = y
        self._current_x = x + 1

    def addstr(self, y, x, msg, attr=None):
        """
        See `Window.addstr()`.
        """
        for i, ch in enumerate(msg):
            self._addch(y=y, x=x + i, ch=ch, attr=attr)

    def erase(self):
        """
        See `Window.erase()`.
        """
        self._data.fill(0)

    def fill_screen(self, screen):
        """
        Fill the screen with pixels.

        :param screen: A numpy array representing the screen buffer (RGB pixels).
        """
        start_y = self._begin_y * _CELL_PIXEL_HEIGHT
        start_x = self._begin_x * _CELL_PIXEL_WIDTH
        stop_y = start_y + self._pixels.shape[0]
        stop_x = start_x + self._pixels.shape[1]
        screen[start_y:stop_y, start_x:stop_x, :] = self._pixels

    def getch(self):
        """
        See `Window.getch()`.
        """
        stop_time = None if self._block_timeout is None else time.perf_counter() + self._block_timeout
        while not _COMMANDS:
            if stop_time is None:
                time.sleep(_DEFAULT_SLEEP_PERIOD)
            elif time.perf_counter() >= stop_time:
                break
            else:
                # noinspection PyTypeChecker
                time.sleep(max(0, stop_time - time.perf_counter()))
        if _COMMANDS:
            return _COMMANDS.popleft()
        else:
            # return KEY_RIGHT
            return -1

    def getmaxyx(self):
        """
        See `Window.getmaxyx()`.
        :return:
        """
        return self._n_lines, self._n_columns

    def move(self, new_y, new_x):
        """
        See `Window.move()`.
        """
        self._current_y = new_y
        self._current_x = new_x

    def noutrefresh(self):
        """
        See `Window.noutrefresh()`.
        """
        self._pixels.fill(0)
        pos_y = 0
        for y in range(self._n_lines):
            pos_x = 0
            for x in range(self._n_columns):
                item_idx = self._data[y, x]
                if item_idx > 0:
                    item = self._items[item_idx]
                    # print(pos_y, pos_x, item_idx)
                    self._pixels[pos_y:pos_y + _CELL_PIXEL_HEIGHT, pos_x:pos_x + _CELL_PIXEL_WIDTH, :] = item
                pos_x += _CELL_PIXEL_WIDTH
            pos_y += _CELL_PIXEL_HEIGHT

    def timeout(self, delay):
        """
        See `Window.timeout()`.
        """
        if delay < 0:
            self._block_timeout = None
        else:
            self._block_timeout = delay / 1000.


def _get_rgb_array(color_number):
    """
    Return the RGB numpy array (of dtype `uint8`) associated to the given color number.
    """
    try:
        return _COLOR_TO_RGB[color_number]
    except KeyError:
        rgb = np.array(color_content(color_number), dtype=np.float64)
        rgb *= _MAX_RGB_VALUE / _MAX_COLOR_INTENSITY
        rgb = np.minimum(rgb, _MAX_RGB_VALUE).astype(np.uint8)
        _COLOR_TO_RGB[color_number] = rgb
        return rgb


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


def doupdate():
    """
    See `curses.doupdate()`.
    """
    _PIXELS.fill(0)
    for window in _ALL_WINDOWS:
        window.fill_screen(_PIXELS)
    for callback in _CALLBACKS:
        callback()



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
    window = JupyWindow(n_lines=nlines, n_columns=ncols, begin_x=begin_x, begin_y=begin_y)
    _ALL_WINDOWS.append(window)
    return window


def pair_content(pair_number):
    """
    See `curses.pair_content()`.
    """
    return _COLOR_PAIRS[pair_number]


def pixels_to_image():
    """
    Convert raw pixel data to ipywidgets image format.
    """
    with io.BytesIO() as b_out:
        im = PIL.Image.fromarray(_PIXELS, mode='RGB')
        # im = im.resize(size=(width, height), resample=PIL.Image.NEAREST)  # blurry without this step
        im.save(b_out, format=_IMAGE_FORMAT)
        return b_out.getvalue()


def register_callback(func):
    """
    Register a callback to be called whenever the screen is updated.

    :param func: The function to be called (with no parameter).
    """
    _CALLBACKS.append(func)


def replace_curses():
    """
    Register this module as "curses" in `sys.modules`, so that anyone trying to use `curses` will use this one instead.
    """
    sys.modules['curses'] = sys.modules['jupycurses']


def update_image():
    """
    Update and return the `ipywidgets.Image` corresponding to the current state of pixels.
    """
    global _IMAGE
    if _IMAGE is None:
        _IMAGE = ipywidgets.Image(
            value=pixels_to_image(), format=_IMAGE_FORMAT, height=_PIXELS.shape[0], width=_PIXELS.shape[1])
        register_callback(update_image)
    else:
        _IMAGE.value = pixels_to_image()
    return _IMAGE


def wrapper(func):
    """
    See `curses.wrapper()`.
    """
    window = JupyWindow()
    _ALL_WINDOWS.append(window)
    func(window)

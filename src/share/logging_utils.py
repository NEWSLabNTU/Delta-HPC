import logging


class ColoredFormatter(logging.Formatter):
    """Custom logging formatter that adds ANSI color codes to log levels selectively."""

    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    # Selective coloring: levelname and filename:lineno
    FORMAT_STR = "%(asctime)s [{color}%(levelname)s{reset}] {color}%(filename)s:%(lineno)d{reset}: %(message)s \n"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    LEVEL_COLORS = {
        logging.DEBUG: BLUE,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD + RED,
    }

    def __init__(self):
        super().__init__(datefmt=self.DATE_FORMAT)

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        # We must create a new formatter with the pre-colored string because
        # standard logging.Formatter doesn't allow dynamic format strings per record easily
        fmt = self.FORMAT_STR.format(color=color, reset=self.RESET)
        formatter = logging.Formatter(fmt, datefmt=self.DATE_FORMAT)
        return formatter.format(record)


def setup_logging(level: int = logging.INFO):
    """Set up colored logging to the console.

    Parameters
    ----------
    level:
        The logging level to set for the root logger.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())
    logging.root.setLevel(level)
    # Clear existing handlers to avoid duplicate logs if setup_logging is called multiple times
    logging.root.handlers = []
    logging.root.addHandler(handler)

    # Suppress verbose HTTP request logs from openai and httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

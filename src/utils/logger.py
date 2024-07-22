import os
import logging
logger = logging.getLogger("vOPA_logger")


class CustomFormatter(logging.Formatter):

    def __init__(self, fmt_str, style):
        super().__init__()
        self.style = style
        self.fmt_str = fmt_str
        white = "\x1b[37;20m"
        reset = "\x1b[0m"

        self.formats = {logging.DEBUG: white + self.fmt_str + reset, logging.INFO: white + self.fmt_str + reset,
                        logging.WARNING: white + self.fmt_str + reset, logging.ERROR: white + self.fmt_str + reset,
                        logging.CRITICAL: white + self.fmt_str + reset}

    def format(self, record):
        """Format the log"""
        log_fmt = self.formats.get(record.levelno)
        formatter = logging.Formatter(log_fmt, style=self.style)
        return formatter.format(record)


def set_logger(file_name) -> None:
    """Set up file and console handlers for logger"""
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    if len(file_name.split('/')) > 1:
        os.makedirs('/'.join(file_name.split('/')[:-1]), exist_ok=True)
    fh = logging.FileHandler(file_name, mode='w')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create a formatter and add it to the handlers
    file_format = "{levelname:<10} | {module:<10} {funcName:<15} | line {lineno:>3}: {message}\n"
    console_format = "{message}"
    ch.setFormatter(CustomFormatter(console_format, style='{'))
    fh.setFormatter(logging.Formatter(file_format, style='{'))
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

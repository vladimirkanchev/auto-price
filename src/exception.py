"""Custom exception for the project."""
import sys

from logger import logging


def error_message_detail(error, error_detail: sys):
    """Error messages to record into log file."""
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occured in pythоn script name [{0}]" \
        + f"line number [{1}] error message[{2}]".format(
            file_name, exc_tb.tb_lineno, str(error))

    return error_message


class CustomException(Exception):
    """Customized exception for the auto project."""

    def __init__(self, error_message, error_detail: sys):
        """Initialize error message of the customized exception."""
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,
                                                  error_detail=error_detail)

    def __str__(self):
        """Error message of the custom exception."""
        return self.error_message


if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as err:
        logging.info("Divde by Zero")
        raise CustomException(err, sys) from None

import csv
from typing import Any, Dict, Tuple, Union

import settings
import os


class CSVOutput(object):
    def __init__(self, filename: str, overwrite_file: bool, delimiter=',', header=settings.RESULTS_HEADER):
        """
        log to a file, in a CSV format

        :param filename: (str) the file to write the log to
        """

        abs_path = filename = os.path.join(settings.RESULTS_DIR, filename + '.csv')

        mode = 'w' if overwrite_file else 'a'  # use 'w+' or 'a+' if also required to read file

        self.file = open(abs_path, mode)

        self.csv_writer = csv.DictWriter(self.file, delimiter=delimiter, fieldnames=header)
        self.csv_writer.writeheader()
        # self.keys = []

    def write(self, result: Dict[str, Union[str, Tuple[str, ...]]]) -> None:
        """
        writes to  file
        """
        self.csv_writer.writerow(result)

    def close(self) -> None:
        """
        closes the file
        """
        self.file.close()

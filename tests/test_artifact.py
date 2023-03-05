import os
import shutil
import unittest

from artifact import split_datafiles


class DatasetTestCase(unittest.TestCase):
    def setUp(self):
        self.datadir = "./mock-data/artifact-splitdatafiles"
        self.expected_datafiles_index_summary = {
            "train": 41,
            "dev": 5,
            "evaluation": 5,
        }

    def test_split_datafiles(self):
        datafiles_index = split_datafiles(
            datadir=self.datadir,
            train=0.8,
            dev=0.1,
            evaluation=0.1,
        )
        datafiles_index_summary = {
            split_name: len(files_in_split)
            for split_name, files_in_split in datafiles_index.items()
        }

        self.assertEqual(datafiles_index_summary, self.expected_datafiles_index_summary)

    def tearDown(self):
        for split in self.expected_datafiles_index_summary.keys():
            shutil.rmtree(os.path.join(self.datadir, split))


if __name__ == "__main__":
    unittest.main()

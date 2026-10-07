import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mhealth import mhealth_data


class MHealthDataTests(unittest.TestCase):
    def test_feature_reader_never_uses_downstream_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp) / "mhealth" / "data"
            data_dir.mkdir(parents=True)
            np.savetxt(data_dir / "data.txt", [[1, 2, 3], [4, 5, 6]])
            (data_dir / "feature_names.txt").write_text("x\ny\nz\n")
            (data_dir / "labels.txt").write_text("not a numeric label file\n")
            with patch.object(mhealth_data, "DATA_ROOT", Path(temp)):
                np.testing.assert_array_equal(
                    mhealth_data.read_features("mhealth"), [[1, 2, 3], [4, 5, 6]]
                )
                np.savetxt(data_dir / "data.txt", [[1, 2, 3, 9], [4, 5, 6, 8]])
                with self.assertRaisesRegex(ValueError, "4 columns but 3 named features"):
                    mhealth_data.read_features("mhealth")

    def test_bidirectional_edges_reconstruct_feature_cells(self):
        values = np.array([[0.1, 0.2], [0.3, 0.4]])
        edges, attributes = mhealth_data.bipartite_arrays(values)
        np.testing.assert_array_equal(edges, [[0, 0, 1, 1, 2, 3, 2, 3],
                                              [2, 3, 2, 3, 0, 0, 1, 1]])
        np.testing.assert_allclose(attributes[:, 0], [0.1, 0.2, 0.3, 0.4] * 2)


if __name__ == "__main__":
    unittest.main()

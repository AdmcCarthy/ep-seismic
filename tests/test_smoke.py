import numpy as np
import segyio

import ep_seismic.volume as seismic_lib


def test_package_imports() -> None:
    assert seismic_lib.__version__


def test_segyio_roundtrip_synthetic(tmp_path) -> None:
    """segyio can write and read back a tiny synthetic 3D file."""
    path = tmp_path / "tiny.sgy"
    ilines, xlines, samples = [1, 2], [10, 11, 12], np.arange(0, 20, 4.0)
    spec = segyio.spec()
    spec.format = 5
    spec.ilines, spec.xlines, spec.samples = ilines, xlines, samples
    data = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    with segyio.create(path, spec) as f:
        for i, il in enumerate(ilines):
            for j, xl in enumerate(xlines):
                n = i * len(xlines) + j
                f.header[n] = {
                    segyio.TraceField.INLINE_3D: il,
                    segyio.TraceField.CROSSLINE_3D: xl,
                }
                f.trace[n] = data[i, j]
    with segyio.open(path) as f:
        assert f.ilines.tolist() == ilines
        np.testing.assert_array_equal(f.iline[2], data[1])

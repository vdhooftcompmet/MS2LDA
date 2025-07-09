from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from matchms.plotting.spectrum_plots import plot_spectra_mirror, plot_spectrum
from MS2LDA.Visualisation.visualisation import show_annotated_motifs
from matchms import Fragments
from matchms.hashing import metadata_hash, spectrum_hash
from matchms import Metadata
from matchms import set_matchms_logger_level
from matchms import Spectrum

set_matchms_logger_level("CRITICAL")


class Mass2Motif:
    """Container for a collection of peaks, losses and metadata.

    Spectrum peaks are stored as :class:`~matchms.Fragments` object which can be
    addressed calling `spectrum.peaks` and contains m/z values and the respective
    peak intensities.

    Spectrum metadata is stored as :class:`~matchms.Metadata` object which can be
    addressed by `spectrum.metadata`.

    Code example

    .. testcode::

        import numpy as np
        from matchms import Scores, Spectrum
        from matchms.similarity import CosineGreedy

        spectrum = Spectrum(mz=np.array([100, 150, 200.]),
                            intensities=np.array([0.7, 0.2, 0.1]),
                            metadata={"id": 'spectrum1',
                                      "precursor_mz": 222.333,
                                      "peak_comments": {200.: "the peak at 200 m/z"}})

        print(spectrum)
        print(spectrum.peaks.mz[0])
        print(spectrum.peaks.intensities[0])
        print(spectrum.get('id'))
        print(spectrum.peak_comments.get(200))

    Should output

    .. testoutput::

        Spectrum(precursor m/z=222.33, 3 fragments between 100.0 and 200.0)
        100.0
        0.7
        spectrum1
        the peak at 200 m/z

    Attributes
    ----------
    peaks: ~matchms.Fragments.Fragments
        Peaks of spectrum
    losses: ~matchms.Fragments.Fragments or None
        Losses of spectrum, the difference between the precursor and all peaks.

        Can be filled with

        .. code-block ::

            from matchms import Fragments
            spectrum.losess = Fragments(mz=np.array([50.]), intensities=np.array([0.1]))
    metadata: dict
        Dict of metadata with for example the scan number of precursor m/z.

    """

    _peak_comments_mz_tolerance = 1e-05

    def __init__(
        self,
        frag_mz: np.array,
        frag_intensities: np.array,
        loss_mz: np.array,
        loss_intensities: np.array,
        metadata: Optional[dict] = None,
        metadata_harmonization: bool = True,
    ):
        """

        Parameters
        ----------
        mz
            Array of m/z for the peaks
        intensities
            Array of intensities for the peaks
        metadata
            Dictionary with for example the scan number of precursor m/z.
        metadata_harmonization : bool, optional
            Set to False if default metadata filters should not be applied.
            The default is True.
        """
        self._metadata = Metadata(metadata)
        if metadata_harmonization is True:
            self._metadata.harmonize_values()
        self.peaks = Fragments(mz=frag_mz, intensities=frag_intensities)
        self._losses = Fragments(mz=loss_mz, intensities=loss_intensities)

    def __eq__(self, other):
        return self.peaks == other.peaks and self._metadata == other._metadata

    def __hash__(self):
        """Return a integer hash which is computed from both
        metadata (see .metadata_hash() method) and spectrum peaks
        (see .spectrum_hash() method)."""
        combined_hash = self.metadata_hash() + self.spectrum_hash()
        return int.from_bytes(bytearray(combined_hash, "utf-8"), "big")

    def __repr__(self):
        precursor_mz_str = f"{self.get('precursor_mz', 0.0):.2f}"
        num_peaks = len(self.peaks)
        if num_peaks == 0:
            return f"Spectrum(precursor m/z={precursor_mz_str}, no fragments)"
        min_mz = min(self.peaks.mz)
        max_mz = max(self.peaks.mz)
        return f"Spectrum(precursor m/z={precursor_mz_str}, {num_peaks} fragments between {min_mz:.1f} and {max_mz:.1f})"

    def __str__(self):
        return self.__repr__()

    def spectrum_hash(self):
        """Return a (truncated) sha256-based hash which is generated
        based on the spectrum peaks (mz:intensity pairs).
        Spectra with same peaks will results in same spectrum_hash."""
        return spectrum_hash(self.peaks)

    def metadata_hash(self):
        """Return a (truncated) sha256-based hash which is generated
        based on the spectrum metadata.
        Spectra with same metadata results in same metadata_hash."""
        return metadata_hash(self._metadata.data)

    def clone(self):
        """Return a deepcopy of the spectrum instance."""
        clone = Mass2Motif(
            frag_mz=self.peaks.mz,
            frag_intensities=self.peaks.intensities,
            loss_mz=self._losses.mz,
            loss_intensities=self._losses.intensities,
            metadata=self._metadata.data,
            metadata_harmonization=False,
        )
        return clone

    def plot(self, figsize=(8, 6), dpi=200, **kwargs):
        """Plot to visually inspect a spectrum run ``spectrum.plot()``

        .. figure:: ../_static/spectrum-plot-example.png
            :width: 450
            :alt: spectrum plotting function

            Example of a spectrum plotted using ``spectrum.plot()`` ..
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax = plot_spectrum(self, ax=ax, **kwargs)
        return fig, ax

    def plot_against(self, other_spectrum, figsize=(8, 6), dpi=200, **spectrum_kws):
        """Compare two spectra visually in a mirror plot.

        To visually compare the peaks of two spectra run
        ``spectrum.plot_against(other_spectrum)``

        .. figure:: ../_static/spectrum-mirror-plot-example.png
            :width: 450
            :alt: spectrum mirror plot function

            Example of a mirror plot comparing two spectra ``spectrum.plot_against()`` ..
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax = plot_spectra_mirror(self, other_spectrum, ax=ax, **spectrum_kws)
        return fig, ax

    def get(self, key: str, default=None):
        """Retrieve value from :attr:`metadata` dict. Shorthand for

        .. code-block:: python

            val = self.metadata[key]

        """
        return self._metadata.get(key, default)

    def set(self, key: str, value):
        """Set value in :attr:`metadata` dict. Shorthand for

        .. code-block:: python

            self.metadata[key] = val

        """
        self._metadata.set(key, value)
        return self

    def to_dict(self, export_style: str = "matchms") -> dict:
        """Return a dictionary representation of a spectrum.

        Parameters
        ----------
        export_style:
            Converts the keys to the required export style. One of ["matchms", "massbank", "nist", "riken", "gnps"].
            Default is "matchms"
        """
        peaks_list = np.vstack((self.peaks.mz, self.peaks.intensities)).T.tolist()
        spectrum_dict = self.metadata_dict(export_style)  # dict(self.metadata.items())
        spectrum_dict["peaks_json"] = peaks_list
        if "fingerprint" in spectrum_dict:
            spectrum_dict["fingerprint"] = spectrum_dict["fingerprint"].tolist()
        return spectrum_dict

    def metadata_dict(self, export_style: str = "matchms") -> dict:
        """Convert spectrum metadata to Python dictionary.

        Parameters
        ----------
        export_style:
            Converts the keys to the required export style. One of ["matchms", "massbank", "nist", "riken", "gnps"].
            Default is "matchms"
        """
        return self._metadata.to_dict(export_style)

    @property
    def mz(self):
        return self.peaks.mz

    @property
    def intensities(self):
        return self.peaks.intensities

    @property
    def metadata(self):
        return self._metadata.data.copy()

    @metadata.setter
    def metadata(self, value):
        self._metadata.data = value

    @property
    def losses(self) -> Optional[Fragments]:
        return self._losses

    @property
    def peaks(self) -> Fragments:
        return self._peaks.clone()

    @peaks.setter
    def peaks(self, value: Fragments):
        if isinstance(self.get("peak_comments"), dict):
            self._reiterate_peak_comments(value)
        self._peaks = value

    @property
    def peak_comments(self):
        return self.get("peak_comments")

    @peak_comments.setter
    def peak_comments(self, value):
        self.set("peak_comments", value)

    @classmethod
    def update_peak_comments_mz_tolerance(cls, mz_tolerance: float):
        """Change current peak comment m/z tolerance to mz_tolerance."""
        cls._peak_comments_mz_tolerance = mz_tolerance

    def _reiterate_peak_comments(self, peaks: Fragments):
        """Update the peak comments to reflect the new peaks."""
        if not isinstance(self.get("peak_comments", None), dict):
            return None

        self._metadata["peak_comments"] = {
            float(key) if isinstance(key, str) else key: value
            for key, value in self.metadata["peak_comments"].items()
        }

        mz_tolerance = self._peak_comments_mz_tolerance

        def _append_new_comment(key):
            if new_key_comment is not None:
                comment = "; ".join(
                    [new_key_comment, self.metadata["peak_comments"].get(key)]
                )
            else:
                comment = self.metadata["peak_comments"].get(key)
            return comment

        for key in list(self.metadata["peak_comments"].keys()):
            if key not in peaks.mz:
                if np.isclose(key, peaks.mz, rtol=mz_tolerance).any():
                    new_key = peaks.mz[
                        np.isclose(key, peaks.mz, rtol=mz_tolerance).argmax()
                    ]
                    new_key_comment = self.metadata["peak_comments"].get(new_key, None)
                    new_key_comment = _append_new_comment(key)
                    self._metadata["peak_comments"][new_key] = new_key_comment
                self._metadata["peak_comments"].pop(key)

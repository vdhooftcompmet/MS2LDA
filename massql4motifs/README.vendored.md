# Vendored Package: massql4motifs

This directory contains a vendored copy of the massql4motifs package.

## Source Information
- **Repository**: https://github.com/j-a-dietrich/MassQueryLanguage4Mass2Motifs.git
- **Commit**: ac1b10b1f020fdfde114c17d57f0ed9b435d9d47
- **Date vendored**: 2025-07-01
- **Vendored by**: MS2LDA maintainers

## Reason for Vendoring
This package has been vendored to enable distribution of MS2LDA on PyPI. PyPI does not allow packages with direct Git dependencies, so we've included a snapshot of massql4motifs within our package.

## Original Authors
- Jonas Dietrich (jonas.dietrich@wur.nl)
- Rosina Torres Ortega (rosina.torresortega@wur.nl)

## License
This vendored code is distributed under the same license as the original MassQueryLanguage4Mass2Motifs project. See LICENSE.txt in the original repository.

## Future Plans
This vendoring is temporary. When massql4motifs is published to PyPI or when motif support is merged into the upstream MassQL project, we will remove this vendored copy and use the official package instead.

## Updating the Vendored Copy
To update this vendored copy:
1. Clone the source repository at the desired commit
2. Copy the massql4motifs directory to this location
3. Update the commit hash and date in this file
4. Test that all functionality still works
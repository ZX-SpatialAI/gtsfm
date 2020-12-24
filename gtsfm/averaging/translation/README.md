# Translation averaging

Translation averaging is the estimation of the global translation from pairwise translation direction measurements. We follow the approach introduced by [Kyle Wilson and Noah Snavely](https://research.cs.cornell.edu/1dsfm/docs/1DSfM_ECCV14.pdf). In this approach, outliers are first removed by projecting the measurements onto 1 dimension and checking for consistent orderings. This involves solving the Minimum Feedback Arc Set graph problem. In the second stage, the measurements without outliers are used to optimize the global translations using a squared chordal distance cost. 
"""macOS setup: sync dotfiles, install packages, configure macOS, and reverse it all.

`setup.sh` is a shim that runs this package. Every run archives what it changes so a later
``--uninstall`` can restore the prior state (guarded: only reverts values it still owns).
"""

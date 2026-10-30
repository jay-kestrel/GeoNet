# SPDX-License-Identifier: GPL-3.0-or-later
# GeoNetPorts — Julia ports of GeoNet's core numeric functionality.
#
# One submodule per Python stage, mirroring the pygeonet_* module names.
# The subproject is self-contained: kernels read plain arrays / .npy fixtures
# and never import the Python modules.

module GeoNetPorts

include("FastMarching.jl")
using .FastMarching
export FastMarching, travel_time, FastMarchingError

end # module

""" Visualization for Delta Sets and their duals.

This module provides wrapper functions for plotting embedded delta sets
using the Makie functionality. This includes the wireframe, mesh, and scatter
plotting functions.

It is important to note that the end-user will need to import one of the Makie
backends (CairoMakie, GLMakie, or WGLMakie) in order for plotting to work
correctly.
"""

module MeshGraphics

using Catlab.CategoricalAlgebra
using ..SimplicialSets
using ..MeshInterop
#import ...AbstractPlotting: wireframe, wireframe!, mesh, mesh!, scatter, scatter!
#
#export wireframe, wireframe!, mesh, mesh!, scatter, scatter!

import ...AbstractPlotting
import ...AbstractPlotting: convert_arguments

""" This extends the "Mesh" plotting recipe for embedded deltasets by converting
an embedded deltaset into arguments that can be passed into AbstractPlotting.mesh
"""

function convert_arguments(P::Union{Type{<:AbstractPlotting.Wireframe},
                                    Type{<:AbstractPlotting.Mesh},
                                    Type{<:AbstractPlotting.Scatter}},
                           dset::AbstractACSet)
  convert_arguments(P, Mesh(dset))
end

""" This extends the "LineSegments" plotting recipe for embedded deltasets by converting
an embedded deltaset into arguments that can be passed into AbstractPlotting.linesegments
"""
function convert_arguments(P::Type{<:AbstractPlotting.LineSegments}, dset::EmbeddedDeltaSet2D)
  edge_positions = zeros(ne(dset)*2,3)
  for e in edges(dset)
    edge_positions[2*e-1,:] = point(dset, dset[e,:src])
    edge_positions[2*e,:] = point(dset, dset[e,:tgt])
  end
  convert_arguments(P, edge_positions)
end

plottype(::EmbeddedDeltaSet2D) = Mesh

end

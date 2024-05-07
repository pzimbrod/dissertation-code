dim_err = "Dimensionality Error: Dimension of requested topological entity exceeds
dimension of cell."

ndofs(::Interval,::Cell,order)      = order+1
ndofs(::Interval,::Vertex,order)    = order == 0 ? 1 : 2
ndofs(::Interval,::Edge,order)      = max(0,order-1)
ndofs(::Interval,::Facet,order)     = ndofs(Interval(),Vertex(),order)
ndofs(::Interval,::TopologicalEntity,order) = throw(dim_err)

ndofs(::Triangle,::Cell,order)      = Int((order+1)*(order+2)/2)
ndofs(::Triangle,::Vertex,order)    = order == 0 ? 1 : 3
ndofs(::Triangle,::Edge,order)      = max(0,order-1) * 3
ndofs(::Triangle,::Face,order)      = Int((order-1)*(order-2)/2)
ndofs(::Triangle,::Facet,order)     = ndofs(Triangle(),Edge(),order)
ndofs(::Triangle,::TopologicalEntity,order) = throw(dim_err)

ndofs(::Tetrahedron,order)  = Int((order+1)*(order+2)*(order+3)/6)

ndofs(::Quadrilateral,::Cell,order)     = Int((order+1)^2)
ndofs(::Quadrilateral,::Vertex,order)   = order == 0 ? 1 : 4
ndofs(::Quadrilateral,::Edge,order)     = max(0,order-1) * 4
ndofs(::Quadrilateral,::Face,order)     = Int((order-1)^2)
ndofs(::Quadrilateral,::Facet,order)    = ndofs(Quadrilateral(),Edge(),order)
ndofs(::Quadrilateral,::TopologicalEntity,order) = throw(dim_err)

ndofs(::Hexahedron,order)   = Int((order+1)^3)

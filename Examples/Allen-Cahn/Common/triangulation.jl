abstract type Primitive end
abstract type TensorProductPrimitive    <: Primitive end
abstract type Simplex                   <: Primitive end
abstract type CompositePrimitive        <: Primitive end
struct Interval                         <: TensorProductPrimitive end
struct Quadrilateral                    <: TensorProductPrimitive end
struct Hexahedron                       <: TensorProductPrimitive end
struct Triangle                         <: Simplex end
struct Tetrahedron                      <: Simplex end
struct Prism                            <: CompositePrimitive end
struct Pyramid                          <: CompositePrimitive end
primitive_dim(::Interval) = 1
primitive_dim(::Quadrilateral) = 2
primitive_dim(::Triangle) = 2

abstract type Triangulation end

struct FETriangulation{V,C} <: Triangulation
    vertices::V
    connectivity::C
    dim::Int
end

struct FDTriangulation{V,D} <: Triangulation
    h::V
    dim::Int
    domain::D
end

function create_cartesian_grid(x,y,nx,ny)
    return hcat(
        repeat(LinRange(x...,nx),outer=ny),
        repeat(LinRange(y...,ny),inner=nx),
    )
end

function create_cartesian_grid(x,nx)
    return collect(LinRange(x...,nx))
end

function create_cartesian_connectivity(::Quadrilateral,nx,ny)
    # Bottom left vertices only extend to nx-1 for each row
    bottom_left     = setdiff(1:nx*(ny-1)-1,nx:nx:nx*ny)
    bottom_right    = bottom_left .+ 1
    top_left        = bottom_left .+ nx
    top_right       = bottom_right .+ nx
    return hcat(bottom_left,bottom_right,top_left,top_right)
end

function create_cartesian_connectivity(::Interval,nx)
    left    = 1:nx-1
    right   = 2:nx
    return hcat(left,right)
end


function FETriangulation(p::Primitive,x,y,Δx,Δy)
    dim = primitive_dim(p)
    nx, ny = round(Int,(last(x) - first(x)) / Δx)+1, round(Int,(last(y) - first(y)) / Δy)+1
    vertices = create_cartesian_grid(x,y,nx,ny)
    connectivity = create_cartesian_connectivity(p,nx,ny)
    return FETriangulation(
        vertices,
        connectivity,
        dim
    )
end

function FETriangulation(p::Primitive,x,Δx)
    dim = primitive_dim(p)
    nx  = round(Int,(last(x) - first(x)) / Δx)+1
    vertices = create_cartesian_grid(x,nx)
    connectivity = create_cartesian_connectivity(p,nx)
    return FETriangulation(
        vertices,
        connectivity,
        dim
    )
end

function FDTriangulation(x,Δx)
    dim = length(Δx)
    domain = map((x,Δx) -> range(x...,step=Δx),x,Δx)
    return FDTriangulation(
        Δx,
        dim,
        domain
    )
end

FDTriangulation(x,Δx::Number) = FDTriangulation(tuple(x),tuple(Δx))

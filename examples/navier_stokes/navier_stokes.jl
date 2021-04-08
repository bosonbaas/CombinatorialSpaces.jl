using CombinatorialSpaces
using GLMakie
using CairoMakie
using GeometryBasics
using SparseArrays
using LinearAlgebra
using Distributions
using DifferentialEquations
using Catlab.CategoricalAlgebra

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

# Helper functions
gen_form(::Type{Val{1}}, s, vf) = begin
  map(1:ne(s)) do e
    src = s[s[e, :src], :point]
    tgt = s[s[e, :tgt], :point]
    (tgt - src) ⋅ vf((src+tgt)/2) * (s[e,:edge_orientation] == 0 ? 1 : -1) * norm(tgt - src)# * star_1.diag[e]
  end
end
function gen_form(s::EmbeddedDeltaSet2D, f::Function)
  map(f, point(s))
end


# Begin hand-made simulation

k = 0.1 # Diffusion
dt = 0.0025

hand_drawn = false
if(hand_drawn)
  w_m = 100.0
  h_m = 30.0
  scale = 1.5
  w = floor(Int64,100*scale)
  h = floor(Int64,30*scale)
  num_oscillators = w*h
  
  dx0 = 1.5
  dy0 = 1.5
  
  ind(x,y) = x + (y-1)*(w)
  tri_ind(x,y) = x + (y-1)*(w-1)*2
  
  points = Array{Point{3, Float64}}(undef, num_oscillators)
  
  for y in 1:h
    for x in 1:w
      points[ind(x,y)] = Point{3,Float64}((w_m*(x-1))/(w-1),(h_m*(y-1))/(h-1),0)
    end
  end
  
  s = EmbeddedDeltaSet2D{Bool, Point{3,Float64}}()
  add_vertices!(s, num_oscillators, point=points)
  
  for y in 1:(h-1)
      for x in 1:(w-1)
          if (x+y) % 2 == 1
              glue_sorted_triangle!(s, ind(x,y), ind(x+1,y), ind(x,y+1))
              glue_sorted_triangle!(s, ind(x+1,y+1), ind(x,y+1), ind(x+1,y))
          else
              glue_sorted_triangle!(s, ind(x+1,y+1), ind(x,y+1), ind(x,y))
              glue_sorted_triangle!(s, ind(x,y), ind(x+1,y+1), ind(x+1,y))
          end
      end
  end
  # Remove circle from center
  
  circ_x = w_m/2
  circ_y = h_m/2
  circ_r = 2
  rem_points = filter(1:nv(s)) do i
    p = s[i, :point]
    (p[1]-circ_x)^2 + (p[2]-circ_y)^2 <= circ_r^2 && p[2] > circ_y
  end
  rem_edges = unique(vcat(vcat(incident(s, rem_points, :src)...),
                          vcat(incident(s, rem_points, :tgt)...)))
  rem_tris = unique(vcat(vcat(incident(s, rem_edges, :∂e0)...),
                         vcat(incident(s, rem_edges, :∂e1)...),
                         vcat(incident(s, rem_edges, :∂e2)...)))
  
  rem_parts!(s, :Tri, sort(rem_tris))
  bound_edges = unique(vcat(incident(s, rem_points,:src)...,incident(s, rem_points,:tgt)...))
  bound_points = unique(vcat(s[bound_edges,:src],s[bound_edges,:tgt]))
  bound_edges = unique(vcat(incident(s, bound_points,:src)...,incident(s, bound_points,:tgt)...))
  bound_tris  = unique(vcat(incident(s,bound_edges,:∂e0)...,incident(s,bound_edges,:∂e1)...,incident(s,bound_edges,:∂e2)...))
  rem_parts!(s, :E, sort(rem_edges))
  bound_edges = unique(vcat(incident(s, bound_points,:src)...,incident(s, bound_points,:tgt)...))
  #bound_edges = unique(vcat(s[bound_tris,:∂e0],s[bound_tris,:∂e1],s[bound_tris,:∂e2]))
  rem_parts!(s, :V, sort(rem_points))
  bound_points = unique(vcat(s[bound_edges,:src],s[bound_edges,:tgt]))
  
  orient!(s)

else
  s = EmbeddedDeltaSet2D("naca0012_8.stl");
end

min_point =[minimum([p[i] for p in s[:point]]) for i in 1:3]
max_point =[maximum([p[i] for p in s[:point]]) for i in 1:3]
width  = max_point[1] - min_point[1]
height = max_point[2] - min_point[2]

sd = EmbeddedDeltaDualComplex2D{Bool,Float64,Point{3, Float64}}(s);
subdivide_duals!(sd, Barycenter())
fig, ax, ob = wireframe(s, linewidth=0.5)
ax.aspect = AxisAspect(width/height)
save("mesh_viz.png", fig)

# Define transformation functions

# Coboundary operator from 0-simplex to 1-simplex in primal complex
bound_0_1 = d(0, s)
bound_1_2 = d(1, s)
cobound_1_2 = dual_derivative(1, sd)
cobound_0_1 = dual_derivative(0, sd)
star_2 = ⋆(2,sd)
star_1 = ⋆(1,sd)
star_0 = ⋆(0,sd)
inv_star_0 = inv(star_0)
inv_star_1 = -1 .* inv(star_1)
inv_star_2 = inv(star_2)

# Diffusion Physics
C_to_ΔC!(dC, C::Array{Float64}) = begin
    dC .= bound_0_1*C
end


ΔC_to_ϕ!(ϕ, dC) = begin
    ϕ .= star_1*(k.*dC)
end


ϕ_to_∑ϕ!(sϕ, ϕ) = begin
    sϕ .= cobound_1_2*ϕ
end

# Advection Physics
C_to_ΔCv!(ΔCv, v, C) = begin
  ΔCv .= ∧(Tuple{0,1},sd,C,v)
end

ΔCv_to_ϕ!(ϕ, ΔCv) = begin
  ϕ .+= star_1*ΔCv
end


∑ϕ_to_dC!(dC, sϕ) = begin
    dC .= inv_star_0*sϕ
end


dC_to_C!(C, C0, dC) = begin
    C .= C0 .+ dC * dt
end

v_to_u!(u, v) = begin
  u .= star_1 * v
end

u_to_du!(du, v, u) = begin
  du .= (star_1*∧(Tuple{1,0},sd,v,inv_star_0*cobound_1_2*u) .+ cobound_0_1*star_2*∧(Tuple{1,1},sd,v,inv_star_1*u))
  #dp .+= 0.05* (cobound_0_1*star_2*bound_1_2*inv_star_1*p + star_1*bound_0_1*inv_star_0*cobound_1_2*p)
end

du_to_dv!(dv,du) = begin
    dv .= inv_star_1*du
end

dv_to_v!(v, v0, dv) = begin
    v .= v0 .+ dv * dt
end

v_to_dv!(dv, v) = begin
    dv .+= 1* (bound_0_1*inv_star_0*cobound_1_2*star_1*v + inv_star_1*cobound_0_1*star_2*bound_1_2*v)
end


u_to_dp!(dp, u) = begin
  dp .= 30 * inv_star_0 * cobound_1_2*u
end

dp_to_p!(p, p0, dp) = begin
    p .= p0 .+ dp * dt
end

p_to_du!(du, p) = begin
  du .+= -1 .* star_1*bound_0_1*p
end

# Get boundaries

full_bound_edges = findall(x -> abs(x) > 1e-11, boundary(Val{2},s) * fill(1.0,ntriangles(s)))
full_bound_points = unique(vcat(s[full_bound_edges,:src],s[full_bound_edges,:tgt]))

center_mask = findall(x->all((-15.0,-10.0,0.0) .<= x .<= (15.0,10.0,0.0)), s[:point])

edge_points = findall(x->(x[1] == -50.0 || x[1] == 50.0), s[:point])
edge_edges = unique(vcat(incident(s, edge_points,:src)..., incident(s, edge_points,:tgt)...));
edge_points = vcat(s[edge_edges,:src],s[edge_edges,:tgt])
edge_tris  = unique(vcat(incident(s,edge_edges,:∂e0)...,incident(s,edge_edges,:∂e1)...,incident(s,edge_edges,:∂e2)...))
edge_mask = unique(vcat(s[edge_tris,:∂e0],s[edge_tris,:∂e1],s[edge_tris,:∂e2]));
bulk_mask = setdiff(collect(1:nv(s)), edge_points)

roof_floor_points = findall(x->(x[2] == -15.0 || x[2] == 15.0), s[:point])
roof_edges = unique(vcat(incident(s, roof_floor_points,:src)..., incident(s, roof_floor_points,:tgt)...));
roof_floor_points = vcat(s[roof_edges,:src],s[roof_edges,:tgt])
roof_tris  = unique(vcat(incident(s,roof_edges,:∂e0)...,incident(s,roof_edges,:∂e1)...,incident(s,roof_edges,:∂e2)...))
roof_mask = unique(vcat(s[roof_tris,:∂e0],s[roof_tris,:∂e1],s[roof_tris,:∂e2]));


obj_points = filter(i->all((-15,-10,0) .<= s[i,:point] .<= (15,10,0)), full_bound_points)
obj_edges = unique(vcat(incident(s, obj_points,:src)..., incident(s, obj_points,:tgt)...));

# Set intial conditions and run code
velocity(x) = begin
  amp = 2
  amp * Point{3,Float64}(1,0,0)
end
v = gen_form(Val{1}, s, velocity);


values = Dict{Symbol, Array{Float64, 1}}()
values[:IP] =  zeros(Float64, nv(s))  # C
values[:TP] =  zeros(Float64, nv(s))  # dC
values[:IPp] =  zeros(Float64, nv(s))  # p
values[:TPdp] =  zeros(Float64, nv(s))  # dp
values[:IL] =  zeros(Float64, ne(s))  # ΔC
values[:ILCv] =  zeros(Float64, ne(s))  # ΔCu
values[:ILv] =  zeros(Float64, ne(s))  # ΔCu
values[:ISΔv] =  zeros(Float64, ntriangles(s))  # Δv
values[:TLdv] =  zeros(Float64, ne(s))  # ΔCu
values[:IL2u] = zeros(Float64, ne(s))    # ϕ
values[:TL2du] = zeros(Float64, ne(s))    # ∑ϕ
values[:TL2] = zeros(Float64, ne(s))    # ϕ
values[:TV2] = zeros(Float64, nv(s))    # ∑ϕ



function vectorfield!(du, u, t, p)
  values[:IP] .=  0  # C
  values[:TP] .=  0  # dC
  values[:IPp] .=  0  # p
  values[:TPdp] .=  0  # dp
  values[:IL] .=  0  # ΔC
  values[:ILCv] .=  0  # ΔCu
  values[:ILv] .=  0  # ΔCu
  values[:ISΔv] .=  0  # Δv
  values[:TLdv] .=  0  # ΔCu
  values[:IL2u] .= 0    # ϕ
  values[:TL2du] .= 0    # ∑ϕ
  values[:TL2] .= 0   # ϕ
  values[:TV2] .= 0   # ∑ϕ
  
  values[:IP] .= u[1:length(values[:IP])]
  values[:ILv] .= u[(1:length(values[:ILv])).+length(values[:IP])]
  values[:IPp] .= u[(1:length(values[:IPp])).+(length(values[:IP])+length(values[:ILv]))]
  #C_to_ΔC!(values[:IL], values[:IP])
  #C_to_ΔCv!(values[:ILCv], values[:ILv], values[:IP])
  #ΔC_to_ϕ!(values[:TL2], values[:IL])
  #ΔCv_to_ϕ!(values[:TL2],  values[:ILCv])
  #ϕ_to_∑ϕ!(values[:TV2], values[:TL2])
  #∑ϕ_to_dC!(values[:TP], values[:TV2])
  
  # Velocity work
  values[:ILv][vcat(roof_mask)] .= 0
  values[:ILv][obj_edges] .= 0
  
  v_to_u!(values[:IL2u], values[:ILv])
  u_to_du!(values[:TL2du], values[:ILv], values[:IL2u])
  p_to_du!(values[:TL2du], values[:IPp])
  du_to_dv!(values[:TLdv],values[:TL2du])
  v_to_dv!(values[:TLdv],values[:ILv])
  values[:TLdv][vcat(edge_mask, obj_edges, roof_mask)] .= 0
  u_to_dp!(values[:TPdp], values[:IL2u])
  
  du .= vcat(values[:TP], values[:TLdv], values[:TPdp])
end



v = gen_form(Val{1}, s, velocity)
#u = gen_form(Val{1}, s, x->(1/((x[2]-15)^2+(x[1]-15)^2).*(x[2]-15,-(x[1]-15),0)))
tspan = (0.0,40.0)

c = gen_form(s, x->pdf(MultivariateNormal([-20,0],[2.0,2.0]),[x[1],0]))
p = zeros(Float64,nv(s))

println("Beginning Simulation")
prob = ODEProblem(vectorfield!, vcat(c,v,p), tspan)
res = solve(prob, Tsit5(), progress=true, progress_steps=1)

# Plot Results
CairoMakie.activate!()
t = tspan[2]
key = :IPp
f = Figure(resolution = (width*10, height*10))
ax = Axis(f[1, 1])
colors = res(t)[(1:length(values[:IPp])) .+ (length(values[:IP])+length(values[:ILv]))]
mesh!(s, color=colors, colorrange=(minimum(colors[center_mask]),maximum(colors[center_mask])))
ax.aspect = AxisAspect(width/height)
xlims!(ax, -50,50)
ylims!(ax, -15,15)
#wireframe!(s, color=:gray, linewidth=0.5)
save("pressure_crosssection.png", f)

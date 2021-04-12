using Catlab.Graphs
using Catlab.Graphics
using Catlab.CategoricalAlgebra
using CombinatorialSpaces
using GeometryBasics
using Distributions
using CairoMakie
using LinearAlgebra
using DifferentialEquations
include("NewTonti.jl")
using .TontiDiagrams

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

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

s = EmbeddedDeltaSet2D("meshes/naca0012_8.stl");
sd = EmbeddedDeltaDualComplex2D{Bool,Float64,Point{3, Float64}}(s);
subdivide_duals!(sd, Barycenter())

# Get boundary masks for BCs
boundary_e = findall(x -> x != 0, boundary(Val{2},s) * fill(1,ntriangles(s)))
boundary_v = unique(vcat(s[boundary_e,:src],s[boundary_e,:tgt]))

obj_v = filter(x->all((-15.0,-10.0,0.0) .<= s[x,:point] .<= (15.0,10.0,0.0)), boundary_v)
obj_e = unique(vcat(incident(s, obj_v,:src)..., incident(s, obj_v,:tgt)...));

left_v = filter(x->all((-50,-15.0,0.0) .<= s[x,:point] .<= (-50.0,15.0,0.0)), boundary_v)
left_e = unique(vcat(incident(s, left_v,:src)..., incident(s, left_v,:tgt)...));

right_v = filter(x->all((50,-15.0,0.0) .<= s[x,:point] .<= (50.0,15.0,0.0)), boundary_v)
right_e = unique(vcat(incident(s, right_v,:src)..., incident(s, right_v,:tgt)...));

top_v = filter(x->all((-50,15.0,0.0) .<= s[x,:point] .<= (50.0,15.0,0.0)), boundary_v)
top_e = unique(vcat(incident(s, top_v,:src)..., incident(s, top_v,:tgt)...));

bot_v = filter(x->all((-50,-15.0,0.0) .<= s[x,:point] .<= (50.0,-15.0,0.0)), boundary_v)
bot_e = unique(vcat(incident(s, bot_v,:src)..., incident(s, bot_v,:tgt)...));

center_v = filter(x->all((-15.0,-10.0,0.0) .<= s[x,:point] .<= (15.0,10.0,0.0)), 1:nv(s))

color = fill(1.0,nv(s))
color[obj_v] .= 0
fig, ax, ob = mesh(s, color=color)
@show obj_v
save("obj_boundary.png", fig)

# Define DEC operators (will be moved to TontiDiagram tooling)
bound_0_1 = d(0, s)
bound_1_2 = d(1, s)
cobound_1_2 = dual_derivative(1, sd)
cobound_0_1 = dual_derivative(0, sd)
star_2 = ⋆(2,sd)
star_1 = ⋆(1,sd)
star_0 = ⋆(0,sd)
inv_star_0 = inv(star_0)
inv_star_1 = -1 .* inv(star_1)
inv_star_2 = inv(star_2);

td = TontiDiagrams.TontiDiagram()

# Diffusion
TontiDiagrams.add_variables!(td, [:C,:ϕ,:∑ϕ, :∂C, :v, :Cv], [0,1,2,0,1,1], [true, false, false, true, true, true])
TontiDiagrams.add_derivatives!(td, s, sd, :ϕ=>:∑ϕ)
TontiDiagrams.add_time_dep!(td, :∂C, :C)
TontiDiagrams.add_laplacian!(td, sd, :C, :∂C; coef=0.1)

k = 0.1
TontiDiagrams.add_transition!(td, [:∑ϕ], (x,y)->(x.=inv_star_0*y) ,[:∂C])

TontiDiagrams.add_transition!(td, [:C, :v], (Cv,C,v)->(Cv .= ∧(Tuple{0,1},sd,C,v)) ,[:Cv])
TontiDiagrams.add_transition!(td, [:Cv], (x,y)->(x.=star_1*y) ,[:ϕ])

# Flow
TontiDiagrams.add_variables!(td, [:u,:∂u, :∂v], [1,1,1], [false, false, true])
TontiDiagrams.add_transition!(td, [:v], (u,v)->(u .= star_1*v) ,[:u])
TontiDiagrams.add_transition!(td, [:u,:v],
  (∂u,u,v)->((star_1*∧(Tuple{1,0},sd,v,inv_star_0*cobound_1_2*u) .+ cobound_0_1*star_2*∧(Tuple{1,1},sd,v,inv_star_1*u)))
  ,[:∂u])
TontiDiagrams.add_transition!(td, [:∂u], (∂v, ∂u)->(∂v .= inv_star_1*∂u) ,[:∂v])
TontiDiagrams.add_transition!(td, [:v], (∂v,v)->(∂v .= bound_0_1*inv_star_0*cobound_1_2*star_1*v + inv_star_1*cobound_0_1*star_2*bound_1_2*v),[:∂v])
TontiDiagrams.add_time_dep!(td, :∂v, :v)

TontiDiagrams.add_bc!(td, :∂v, v->(v[vcat(left_e,right_e)].=0))
TontiDiagrams.add_bc!(td, :v, v->(v[vcat(top_e,bot_e, obj_e)].=0))

# Pressure

TontiDiagrams.add_variables!(td, [:p,:∂p], [0,0],[true, true])
TontiDiagrams.add_time_dep!(td, :∂p, :p)
TontiDiagrams.add_transition!(td, [:u],  (∂p,u)->(∂p .= 30 * inv_star_0 * cobound_1_2*u), [:∂p])
TontiDiagrams.add_transition!(td, [:p],  (∂u,p)->(∂u .= -1 .* star_1*bound_0_1*p), [:∂u])

data, sim = TontiDiagrams.vectorfield!(td, s);
@show data

c_range = range(data[:C]...,step=1)
v_range = range(data[:v]...,step=1)
p_range = range(data[:p]...,step=1)
u = zeros(Float64,maximum(last.(values(data))))

c = gen_form(s, x->pdf(MultivariateNormal([-20,0],[4.0,2.0]),[x[1],0]))
p = gen_form(s, x->0.0)

velocity(x) = begin
  amp = 1.0 
  amp * Point{3,Float64}(1,0,0)
end
v = gen_form(Val{1}, s, velocity);

u[c_range] .= c
u[v_range] .= v
u[p_range] .= p

tspan=(0.0,10.0)
prob = ODEProblem(sim, u, tspan)
sol = solve(prob, Tsit5(), progress=true, progress_steps=1);

fig, ax, ob = mesh(s, color=sol(1)[1:nv(s)])
save("res.svg",fig)

t = tspan[2]
times = range(0,tspan[2], length=150)
colors = [sol(t)[c_range] for t in times]
figure, axis, scatter_thing = mesh(s, color=colors[1],
                                   colorrange=(minimum(vcat(colors...)),
                                               maximum(vcat(colors...))))
axis.aspect = AxisAspect(100.0/30.0)
framerate = 30

record(figure, "flow_conc.gif", collect(1:length(collect(times))); framerate = framerate) do i
  scatter_thing.color = colors[i]
end


times = range(0,tspan[2], length=150)
colors = [sol(t)[p_range] for t in times]

color_range = [sol(t)[p_range][center_v] for t in times]
figure, axis, scatter_thing = mesh(s, color=colors[1],
                                   colorrange=(minimum(vcat(color_range...)),
                                               maximum(vcat(color_range...))))
axis.aspect = AxisAspect(100.0/30.0)
framerate = 30

record(figure, "flow_press.gif", collect(1:length(collect(times))); framerate = framerate) do i
  scatter_thing.color = colors[i]
end

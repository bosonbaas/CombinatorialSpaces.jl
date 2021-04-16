using Catlab.Graphs
using Catlab.Graphics
using Catlab.CategoricalAlgebra
using CombinatorialSpaces
using GeometryBasics
using Distributions
using CairoMakie
using LinearAlgebra
using DifferentialEquations
#using PProf
include("NewTonti.jl")
using .TontiDiagrams

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

function gen_form(s::EmbeddedDeltaSet2D, f::Function)
  map(f, point(s))
end

s = EmbeddedDeltaSet2D("meshes/naca0012_8.stl");
sp = Space(s)
sd = sp.sd

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
add_variables!(td, (:C,0,true),(:ϕ,1,false),(:∑ϕ,2,false), (:∂C,0,true),
                   (:v,1,true), (:Cv,1,true))
add_derivatives!(td, sp, :ϕ=>:∑ϕ)
add_time_dep!(td, :∂C, :C)
add_laplacian!(td, sp, :C, :∂C; coef=0.1)

add_transition!(td, [:∑ϕ], (x,y)->(x.=inv_star_0*y) ,[:∂C])

add_transition!(td, [:C, :v], (Cv,C,v)->(Cv .= ∧(Tuple{0,1},sd,C,v)) ,[:Cv])
add_transition!(td, [:Cv], (x,y)->(x.=star_1*y) ,[:ϕ])

# Flow
add_variables!(td, (:u,1,false),(:∂u,1,false), (:∂v,1,true))
add_transition!(td, [:v], (u,v)->(u .= star_1*v) ,[:u])
add_transition!(td, [:u,:v],
  (∂u,u,v)->( ∂u .= (star_1*∧(Tuple{1,0},sd,v,inv_star_0*cobound_1_2*u) .+ cobound_0_1*star_2*∧(Tuple{1,1},sd,v,inv_star_1*u)))
  ,[:∂u])
add_transition!(td, [:∂u], (∂v, ∂u)->(∂v .= inv_star_1*∂u) ,[:∂v])
lap_arr = bound_0_1*inv_star_0*cobound_1_2*star_1 + inv_star_1*cobound_0_1*star_2*bound_1_2
#add_transition!(td, [:v], (∂v,v)->(∂v .= lap_arr*v),[:∂v])
add_laplacian!(td, sp, :v, :∂v; coef = 1)
add_time_dep!(td, :∂v, :v)

add_bc!(td, :∂v, v->(v[vcat(left_e,right_e)].=0))
add_bc!(td, :v, v->(v[vcat(top_e,bot_e, obj_e)].=0))

# Pressure

add_variables!(td, (:p,0,true),(:∂p,0,true))
add_time_dep!(td, :∂p, :p)
pressure_op = 30 * inv_star_0 * cobound_1_2
add_transition!(td, [:u],  (∂p,u)->(∂p .= pressure_op*u), [:∂p])
depress_op = -1 .* star_1*bound_0_1
add_transition!(td, [:p],  (∂u,p)->(∂u .= depress_op*p), [:∂u])


data, sim = vectorfield(td, sp);
@show data

c_range = range(data[:C]...,step=1)
v_range = range(data[:v]...,step=1)
p_range = range(data[:p]...,step=1)
u = zeros(Float64,maximum(last.(values(data))))

c = gen_form(s, x->pdf(MultivariateNormal([-25,0],[4.0,2.0]),[x[1],0]))
p = gen_form(s, x->0.0)

velocity(x) = begin
  amp = 2.0 
  amp * Point{3,Float64}(-1,0,0)
end
v = ♭(sd, DualVectorField(velocity.(sd[triangle_center(sd),:dual_point])))

u[c_range] .= c
u[v_range] .= v
u[p_range] .= p

tspan=(0.0,15.0)
prob = ODEProblem(sim, u, tspan)
sol = solve(prob, Tsit5(), progress=true, progress_steps=1);
#@pprof solve(prob, Tsit5(), progress=true, progress_steps=1);

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
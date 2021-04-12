module TontiDiagrams
  using Catlab.Present
  using Catlab.Theories
  using Catlab.CategoricalAlgebra
  using Catlab.CategoricalAlgebra.FinSets
  using Catlab.Graphs
  import Catlab.Graphs: Graph
  using Catlab.Programs
  using Catlab.WiringDiagrams
  using CombinatorialSpaces
  
  @present TheoryTontiDiagram(FreeSchema) begin
    Func::Data
    Comp::Data
    Dimension::Data
    Label::Data

    V::Ob
    I::Ob
    O::Ob
    T::Ob
    BC::Ob
    TD::Ob
  
    iv::Hom(I,V)
    it::Hom(I,T)
    ov::Hom(O,V)
    ot::Hom(O,T)
    bcv::Hom(BC,V)
    deriv::Hom(TD,V)
    integ::Hom(TD,V)

    tfunc::Attr(T,Func)
    bcfunc::Attr(BC,Func)

    complex::Attr(V,Comp)
    dimension::Attr(V,Dimension)
    symbol::Attr(V, Label)
  end

  # These functions are necessary for defining a TontiDiagram data structure from
  # the ACSet
  const AbstractTontiDiagram = AbstractACSetType(TheoryTontiDiagram)
  const TontiDiagram = ACSetType(TheoryTontiDiagram,index=[:iv, :it, :ov, :ot, :bcv],
                                                   unique_index=[:symbol])
  
  # Define an interface for an OpenTontiDiagram which allows Tonti diagrams to
  # be composed over their corners
  const OpenTontiDiagramOb, OpenTontiDiagram = OpenACSetTypes(TontiDiagram,
                                                              :V)
  #==
  Open(td::AbstractTontiDiagram) = begin
    OpenTontiDiagram{Function, Symbol}(td,
                                       FinFunction(collect(1:nparts(td, :Corner)),
                                                   nparts(td, :Corner)))
  end==#

  TontiDiagram() = TontiDiagram{Function, Bool, Int64, Symbol}()
  
  # This can later be used to pre-compute the boundary and hodge-star
  # operators, so that we only need one of each for computation
  TontiDiagram(s::EmbeddedDeltaSet2D, sd) = begin
    TontiDiagram()  
  end

  function vectorfield!(td, s)
    v_mem, t_mem = init_mem(td, s)
    dg = Graph(td)

    order = topological_sort(dg)

    state_vars = filter(x->length(incident(td,x,:ov))==0, 1:nparts(td, :V))

    input_vars = Dict{Symbol, Tuple{Int64,Int64}}()
    cur_head = 1
    for i in state_vars
      v_size = length(v_mem[i])
      input_vars[td[i,:symbol]] = (cur_head,cur_head+v_size-1)
      cur_head += v_size
    end

    function system(du, u, t, p)
      for cur in order
        # Check if current is a variable or transition
        if cur > nparts(td, :V)
          cur -= nparts(td, :V)
          inputs = td[incident(td, cur, :it), :iv]
          outputs = incident(td, cur, :ot)
          td[cur,:tfunc](t_mem[outputs]..., v_mem[inputs]...)
        else
          inputs = incident(td, cur, :ov)
          if(length(inputs) == 0)
            # This means this is a state variable
            data_source = input_vars[td[cur,:symbol]]
            v_mem[cur] .= u[data_source[1]:data_source[2]]
          else
            v_mem[cur] .= 0
            for i in inputs
              v_mem[cur] .+= t_mem[i]
            end
          end
          bcs = incident(td, cur, :bcv)
          for bc in bcs
            td[bc, :bcfunc](v_mem[cur])
          end
        end
      end
      # If a state variable does not have a derivative defined, we keep it out 
      # (we'll want to move these to the parameter argument instead)
      du .= 0
      for i in state_vars
        state_range = input_vars[td[i,:symbol]]
        out_var = td[incident(td, i, :integ), :deriv]
        if length(out_var) != 0
          du[state_range[1]:state_range[2]] .= v_mem[out_var[1]]
        end
      end
    end
    input_vars, system
  end

  function add_variables!(td, symbols::Array{Symbol},
                          dimensions::Array{Int64}, complices::Array{Bool})
    add_parts!(td, :V, length(symbols),symbol=symbols, dimension=dimensions, complex=complices)
  end

  function add_transition!(td, dom_sym::Array{Symbol,1}, func, codom_sym::Array{Symbol,1})
    dom = [findfirst(v->v == s, td[:symbol]) for s in dom_sym]
    codom = [findfirst(v->v == s, td[:symbol]) for s in codom_sym]
    t = add_part!(td, :T, tfunc=func)
    add_parts!(td, :I, length(dom), iv=dom, it=t)
    add_parts!(td, :O, length(codom), ov=codom, ot=t)
  end

  function add_derivative!(td, s, sd, dom_sym, codom_sym)
    dom = findfirst(v->v==dom_sym, td[:symbol])
    codom = findfirst(v->v==codom_sym, td[:symbol])

    # TODO:
    # Add tests for proper dimensions, complexes, etc.
    # This will later be replaced as we pre-initialize all boundary operators
    bound = td[dom,:complex] ? d(td[dom,:dimension],s) : dual_derivative(td[dom,:dimension],sd)
    func(x,y) = (x.=bound*y)
    add_transition!(td, [dom_sym],func,[codom_sym])
  end

  function add_derivatives!(td, s, sd, vars::Pair{Symbol, Symbol}...)
    for v in vars
      add_derivative!(td, s, sd, v[1],v[2])
    end
  end

  function add_time_dep!(td, deriv_sym::Symbol, integ_sym::Symbol)
    deriv = findfirst(v->v==deriv_sym, td[:symbol])
    integ = findfirst(v->v==integ_sym, td[:symbol])

    add_part!(td, :TD, integ=integ, deriv=deriv)
  end

  function add_bc!(td, var_sym, func)
    var = findfirst(v->v==var_sym, td[:symbol])
    add_part!(td, :BC, bcfunc=func, bcv=var)
  end

  # Note: This function can be made more efficient if combined with existing
  # transformations.
  # e.g. Advection-diffusion can be merged after the initial wedge
  # product/coboundary operator
  #
  # Currently only defined on primal complices (can this be applied to dual
  # complices?)
  function add_laplacian!(td, sd, dom_sym, codom_sym; coef=1)
    dom = findfirst(v->v==dom_sym, td[:symbol])
    codom = findfirst(v->v==codom_sym, td[:symbol])

    lap_op = laplace_beltrami(Val{td[dom,:dimension]},sd)
    func(x,y) = (x .= coef * (lap_op*y))
    add_transition!(td, [dom_sym], func, [codom_sym])
  end

  function init_mem(td, s::EmbeddedDeltaSet1D)
    # Fill out this function
  end

  function init_mem(td, s::EmbeddedDeltaSet2D)
    primal_size = [nv(s), ne(s), ntriangles(s)]
    dual_size   = [ntriangles(s), ne(s), nv(s)]

    t_mem = Array{Array{Float64,1},1}()
    v_mem = Array{Array{Float64,1},1}()

    for i in 1:nparts(td, :O)
      var = td[i,:ov]
      push!(t_mem, zeros(Float64, 
                        td[var,:complex] ? primal_size[td[var,:dimension]+1] : 
                        dual_size[td[var,:dimension]+1]))
    end

    for v in 1:nparts(td,:V)
      push!(v_mem, zeros(Float64, 
                        td[v,:complex] ? primal_size[td[v,:dimension]+1] :
                        dual_size[td[v,:dimension]+1]))
    end
    v_mem, t_mem
  end

  function Graph(td)
    g = Graph()
    add_vertices!(g, nparts(td, :V) + nparts(td, :T))
    nvars = nparts(td, :V)
    for i in 1:nparts(td, :I)
      add_edge!(g, td[i,:iv], td[i,:it] + nvars)
    end
    for o in 1:nparts(td, :O)
      add_edge!(g, td[o,:ot] + nvars, td[o,:ov])
    end
    g
  end
end

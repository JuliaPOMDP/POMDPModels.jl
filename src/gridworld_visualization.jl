function render(mdp::SimpleGridWorld, step::Union{NamedTuple,Dict};
                color = s->reward(mdp, s)
               )

    nx, ny = mdp.size
    cells = []
    for x in 1:nx, y in 1:ny
        clr = tocolor(color(GWPos(x,y)))
        ctx = cell_ctx((x,y), mdp.size)
        cell = compose(ctx, rectangle(), fill(clr))
        push!(cells, cell)
    end
    grid = compose(context(), linewidth(0.5mm), stroke("gray"), cells...)
    outline = compose(context(), linewidth(1mm), rectangle())

    if haskey(step, :s)
        agent_ctx = cell_ctx(step[:s], mdp.size)
        agent = compose(agent_ctx, circle(0.5, 0.5, 0.4), fill("orange"))
    else
        agent = nothing
    end
    
    sz = min(w,h)
    return compose(context((w-sz)/2, (h-sz)/2, sz, sz), agent, grid, outline)
end

function cell_ctx(xy, size)
    nx, ny = size
    x, y = xy
    return context((x-1)/nx, (ny-y)/ny, 1/nx, 1/ny)
end

tocolor(x) = x
function tocolor(r::Float64)
    minr = -10.0
    maxr = 10.0
    frac = (r-minr)/(maxr-minr)
    return get(ColorSchemes.redgreensplit, frac)
end

# Autograder.jl -- lightweight autograder for PS5: Autoencoders
#
# Rubric scale (manual discussion review is separate)
#   0 = nothing runs
#   1 = errors, but at least one test passes
#   2 = errors, but most tests pass (at least one error remains)
#   3 = all tests pass, code runs
#   4 = all tests pass/code runs + manual discussion review is good

# Grader holds the running list of test results as named tuples.
# Each entry records: which problem (e.g. "Task 1"), a short description,
# the points earned on that check, and the max points possible.
mutable struct Grader
    results :: Vector{NamedTuple{(:problem,:description,:earned,:total),
                                  Tuple{String,String,Int,Int}}}
end

# Default constructor -- start with an empty results vector
Grader() = Grader(NamedTuple{(:problem,:description,:earned,:total),
                               Tuple{String,String,Int,Int}}[])

"""
    check!(grader, problem, description, pts, testfn)

Run `testfn()` and award `pts` if it returns `true`.  Re-running a check cell
after fixing code *updates* the existing entry rather than duplicating it, so
the score always reflects the most recent run.
"""
function check!(g::Grader, problem::String, description::String,
                pts::Int, testfn::Function)

    # look for an existing entry with the same problem+description key
    # so re-running a cell overwrites the old result instead of appending a duplicate
    idx = findfirst(r -> r.problem == problem &&
                         r.description == description, g.results)
    try
        passed = (testfn() === true)           # run the test; only `true` counts as passing
        earned = passed ? pts : 0              # full credit or zero
        entry  = (problem=problem, description=description, earned=earned, total=pts)

        # insert new or update existing entry
        isnothing(idx) ? push!(g.results, entry) : (g.results[idx] = entry)

        sym = passed ? "✓" : "✗"
        @printf("  %s  %2d / %2d pts  %s\n", sym, earned, pts, description)
    catch e
        # if the test function throws, record zero and print the error type
        entry = (problem=problem, description=description, earned=0, total=pts)
        isnothing(idx) ? push!(g.results, entry) : (g.results[idx] = entry)
        @printf("  ✗   0 / %2d pts  %s  [ERROR: %s]\n", pts, description, typeof(e))
    end
end

"""
    score!(grader)

Print a summary table of earned vs total points, grouped by problem.
"""
function score!(g::Grader; discussion_ok::Bool=false)

    # guard: nothing to report if no checks have been run
    isempty(g.results) && (println("No checks run yet."); return)

    # aggregate totals across all checks
    e_total  = sum(r.earned for r in g.results)  # total earned points
    t_total  = sum(r.total  for r in g.results)  # total possible points
    n_pass   = count(r -> r.earned == r.total, g.results) # number of checks that passed
    n_total  = length(g.results)                 # total number of checks
    n_fail   = n_total - n_pass
    bar_w    = 12                                # width (in chars) of the progress bar

    # print the per-problem summary table
    println()
    println("═"^56)
    @printf("  %-22s  %8s  %s\n", "Problem", "Score", "Progress")
    println("─"^56)

    # group results by problem label and print one row per problem
    for p in unique(r.problem for r in g.results)
        rows   = filter(r -> r.problem == p, g.results)
        pe     = sum(r.earned for r in rows)               # earned for this problem
        pt     = sum(r.total  for r in rows)               # possible for this problem
        filled = round(Int, (pe / max(pt, 1)) * bar_w)    # how many filled blocks
        bar    = "█"^filled * "░"^(bar_w - filled)         # visual progress bar
        @printf("  %-22s  %3d / %3d  [%s]\n", p, pe, pt, bar)
    end

    println("─"^56)
    @printf("  %-22s  %3d / %3d\n", "AUTO-GRADED TOTAL", e_total, t_total)
    println("═"^56)

    # map the pass/fail counts onto the 0-4 rubric scale
    rubric_score = if n_total == 0 || n_pass == 0
        0                                       # nothing works
    elseif n_fail > 0 && n_pass >= 1
        n_pass > (n_total ÷ 2) ? 2 : 1         # some vs most tests pass
    else
        discussion_ok ? 4 : 3                   # all pass; 4 requires manual review
    end

    @printf("  RUBRIC SCORE: %d / 4\n", rubric_score)
    @printf("  Tests passed: %d / %d\n", n_pass, n_total)
    @printf("  Discussion review status: %s\n", discussion_ok ? "approved" : "pending/manual")
    println("  Note: discussion answers are reviewed manually by instructors.")
    println("═"^56)
end
